import sys

sys.path.append("..")

import copy
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from hesiod import get_cfg_copy, get_out_dir, get_run_name, hcfg, hmain
from pycarus.datasets.utils import get_shape_net_category_name_from_id
from pycarus.metrics.partseg_iou import PartSegmentationIoU
from pycarus.utils import progress_bar
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset

from models.idecoder import ImplicitDecoder
from utils import get_class_to_parts

logging.disable(logging.INFO)
os.environ["WANDB_SILENT"] = "true"


class InrEmbeddingDataset(Dataset):
    def __init__(self, root: Path, split: str) -> None:
        super().__init__()

        self.root = root / split
        self.item_paths = sorted(self.root.glob("*.h5"), key=lambda x: int(x.stem))

    def __len__(self) -> int:
        return len(self.item_paths)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        with h5py.File(self.item_paths[index], "r") as f:
            pcd = torch.from_numpy(np.array(f.get("pcd")))
            embedding = np.array(f.get("embedding"))
            embedding = torch.from_numpy(embedding)
            class_id = np.array(f.get("class_id"))
            class_id = torch.from_numpy(class_id).long()
            part_labels = np.array(f.get("part_labels"))
            part_labels = torch.from_numpy(part_labels).long()

        return pcd, embedding, class_id, part_labels


class InrEmbeddingPartSegmentation:
    def __init__(self) -> None:
        dset_root = Path(hcfg("dset_root", str))

        dset_name = hcfg("dset_name", str)
        self.class_to_parts = get_class_to_parts(dset_name)

        train_split = hcfg("train_split", str)
        train_dset = InrEmbeddingDataset(dset_root, train_split)

        train_bs = hcfg("train_bs", int)
        self.train_loader = DataLoader(train_dset, batch_size=train_bs, num_workers=8, shuffle=True)

        val_bs = hcfg("val_bs", int)
        val_split = hcfg("val_split", str)
        val_dset = InrEmbeddingDataset(dset_root, val_split)
        self.val_loader = DataLoader(val_dset, batch_size=val_bs, num_workers=8)

        test_split = hcfg("test_split", str)
        test_dset = InrEmbeddingDataset(dset_root, test_split)
        self.test_loader = DataLoader(test_dset, batch_size=val_bs, num_workers=8)

        self.num_classes = hcfg("num_classes", int)
        self.num_part = hcfg("num_part", int)

        encoder_cfg = hcfg("encoder", Dict[str, Any])
        decoder_cfg = hcfg("decoder", Dict[str, Any])
        decoder = ImplicitDecoder(
            encoder_cfg["embedding_dim"] + self.num_classes,
            decoder_cfg["input_dim"],
            decoder_cfg["hidden_dim"],
            decoder_cfg["num_hidden_layers_before_skip"],
            decoder_cfg["num_hidden_layers_after_skip"],
            self.num_part,
        )
        self.decoder = decoder.cuda()

        lr = hcfg("lr", float)
        wd = hcfg("wd", float)
        self.optimizer = AdamW(self.decoder.parameters(), lr, weight_decay=wd)
        num_steps = hcfg("num_epochs", int) * len(self.train_loader)
        self.scheduler = OneCycleLR(self.optimizer, lr, total_steps=num_steps)

        self.epoch = 0
        self.global_step = 0
        self.best_acc = 0.0

        self.ckpts_path = get_out_dir() / "ckpts"

        if self.ckpts_path.exists():
            self.restore_from_last_ckpt()

        self.ckpts_path.mkdir(exist_ok=True)

    def logfn(self, values: Dict[str, Any]) -> None:
        wandb.log(values, step=self.global_step, commit=False)

    def get_one_hot_encoding(self, x: Tensor, num_classes: int) -> Tensor:
        one_hot = torch.eye(num_classes)[x.cpu()]
        one_hot = one_hot.to(x.device)
        return one_hot

    def train(self) -> None:
        num_epochs = hcfg("num_epochs", int)
        start_epoch = self.epoch

        for epoch in range(start_epoch, num_epochs):
            self.epoch = epoch

            self.decoder.train()

            desc = f"Epoch {epoch}/{num_epochs}"
            for batch in progress_bar(self.train_loader, desc=desc):
                pcds, embeddings, class_labels, part_labels = batch
                embeddings = embeddings.cuda()
                class_labels = class_labels.cuda()
                part_labels = part_labels.cuda()
                pcds = pcds.cuda()
                class_labels = self.get_one_hot_encoding(class_labels, self.num_classes)

                embeddings = torch.cat([embeddings, class_labels], dim=1)
                pred = self.decoder(embeddings, pcds)

                pred = pred.reshape(-1, self.num_part)
                loss = F.cross_entropy(pred, part_labels.reshape(-1))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                if self.global_step % 10 == 0:
                    self.logfn({"train/loss": loss.item()})
                    self.logfn({"train/lr": self.scheduler.get_last_lr()[0]})

                self.global_step += 1

            if epoch % 5 == 0 or epoch == num_epochs - 1:
                self.val("train")
                self.val("val")
                self.save_ckpt()

            if epoch == num_epochs - 1:
                self.val("test", best=True)

    @torch.no_grad()
    def val(self, split: str, best: bool = False) -> None:
        seg_label_to_cat = {}

        for cat in self.class_to_parts:
            for label in self.class_to_parts[cat]:
                seg_label_to_cat[label] = cat

        metric = PartSegmentationIoU(
            use_only_category_logits=True,
            category_to_parts_map=self.class_to_parts,
        )
        metric.reset()

        if split == "train":
            loader = self.train_loader
        elif split == "val":
            loader = self.val_loader
        else:
            loader = self.test_loader

        if best:
            model = self.best_model
        else:
            model = self.decoder

        model = model.to("cuda")
        model.eval()
        losses = []
        for batch in progress_bar(loader, desc=f"Validating on {split} set"):
            pcds, embeddings, class_labels, part_labels = batch
            embeddings = embeddings.cuda()
            class_labels = class_labels.cuda()
            part_labels = part_labels.cuda()
            pcds = pcds.cuda()
            class_labels = self.get_one_hot_encoding(class_labels, self.num_classes)

            embeddings = torch.cat([embeddings, class_labels], dim=1)
            seg_pred = self.decoder(embeddings, pcds)

            out = seg_pred.reshape(-1, self.num_part)
            loss = F.cross_entropy(out, part_labels.reshape(-1))
            losses.append(loss)
            metric.update(seg_pred, part_labels)

        mIoU_per_cat, class_avg_iou, instance_avg_iou = metric.compute()
        self.logfn({f"{split}/class_avg_iou": class_avg_iou})
        self.logfn({f"{split}/instance_avg_iou": instance_avg_iou})
        self.logfn({f"{split}/loss": torch.mean(torch.tensor(losses))})

        if split == "test":
            table = wandb.Table(columns=["class", "ioU"])
            for cat, miou in mIoU_per_cat.items():
                table.add_data(get_shape_net_category_name_from_id(cat), miou)
            wandb.log({"class IoU": table})

        if class_avg_iou > self.best_acc and split == "val":
            self.best_acc = class_avg_iou
            self.save_ckpt(best=True)
            self.best_model = copy.deepcopy(self.decoder)

    def save_ckpt(self, best: bool = False) -> None:
        ckpt = {
            "epoch": self.epoch,
            "best_acc": self.best_acc,
            "decoder": self.decoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }

        for previous_ckpt_path in self.ckpts_path.glob("*.pt"):
            if "best" not in previous_ckpt_path.name:
                previous_ckpt_path.unlink()

        ckpt_path = self.ckpts_path / f"{self.epoch}.pt"
        torch.save(ckpt, ckpt_path)

        if best:
            ckpt_path = self.ckpts_path / "best.pt"
            torch.save(ckpt, ckpt_path)

    def restore_from_last_ckpt(self) -> None:
        if self.ckpts_path.exists():
            ckpt_paths = [p for p in self.ckpts_path.glob("*.pt") if "best" not in p.name]
            error_msg = "Expected only one ckpt apart from best, found none or too many."
            assert len(ckpt_paths) == 1, error_msg

            ckpt_path = ckpt_paths[0]
            ckpt = torch.load(ckpt_path)

            self.epoch = ckpt["epoch"] + 1
            self.global_step = self.epoch * len(self.train_loader)
            self.best_acc = ckpt["best_acc"]

            self.decoder.load_state_dict(ckpt["decoder"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.scheduler.load_state_dict(ckpt["scheduler"])


run_cfg_file = sys.argv[1] if len(sys.argv) == 2 else None


@hmain(
    base_cfg_dir="cfg/bases",
    template_cfg_file="cfg/partseg.yaml",
    run_cfg_file=run_cfg_file,
    parse_cmd_line=False,
)
def main() -> None:
    wandb.init(
        entity="entity",
        project=f"inr2vec",
        name=get_run_name(),
        dir=get_out_dir(),
        config=get_cfg_copy(),
    )

    trainer = InrEmbeddingPartSegmentation()
    trainer.train()


if __name__ == "__main__":
    main()
