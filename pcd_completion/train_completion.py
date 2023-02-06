import sys

sys.path.append("..")

from pathlib import Path
from random import randint
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from hesiod import get_cfg_copy, get_out_dir, get_run_name, hcfg, hmain
from pycarus.geometry.pcd import random_point_sampling, sample_pcds_from_udfs
from pycarus.metrics.chamfer_distance import chamfer_t
from pycarus.metrics.f_score import f_score
from pycarus.utils import progress_bar
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from models.idecoder import ImplicitDecoder
from models.transfer import Transfer


class InrEmbeddingDataset(Dataset):
    def __init__(self, root: Path, split: str) -> None:
        super().__init__()

        self.root = root / split
        self.item_paths = sorted(self.root.glob("*.h5"), key=lambda x: int(x.stem))

    def __len__(self) -> int:
        return len(self.item_paths)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        with h5py.File(self.item_paths[index], "r") as f:
            incomplete = torch.from_numpy(np.array(f.get("incomplete")))
            complete = torch.from_numpy(np.array(f.get("complete")))
            embedding_incomplete = np.array(f.get("embedding_incomplete"))
            embedding_incomplete = torch.from_numpy(embedding_incomplete)
            embedding_complete = np.array(f.get("embedding_complete"))
            embedding_complete = torch.from_numpy(embedding_complete)

        return incomplete, complete, embedding_incomplete, embedding_complete


class CompletionTrainer:
    def __init__(self) -> None:
        dset_root = Path(hcfg("dset_root", str))

        train_split = hcfg("train_split", str)
        train_dset = InrEmbeddingDataset(dset_root, train_split)

        train_bs = hcfg("train_bs", int)
        self.train_loader = DataLoader(train_dset, batch_size=train_bs, num_workers=8, shuffle=True)

        val_bs = hcfg("val_bs", int)
        val_split = hcfg("val_split", str)
        val_dset = InrEmbeddingDataset(dset_root, val_split)
        self.val_loader = DataLoader(val_dset, batch_size=val_bs, num_workers=8)
        self.train_val_loader = DataLoader(train_dset, batch_size=val_bs, num_workers=8)

        embedding_dim = hcfg("embedding_dim", int)
        num_layers = hcfg("num_layers_transfer", int)
        transfer = Transfer(embedding_dim, num_layers)
        self.transfer = transfer.cuda()

        decoder_cfg = hcfg("decoder", Dict[str, Any])
        decoder = ImplicitDecoder(
            embedding_dim,
            decoder_cfg["input_dim"],
            decoder_cfg["hidden_dim"],
            decoder_cfg["num_hidden_layers_before_skip"],
            decoder_cfg["num_hidden_layers_after_skip"],
            decoder_cfg["out_dim"],
        )
        decoder_ckpt_path = hcfg("decoder_ckpt_path", str)
        decoder_ckpt = torch.load(decoder_ckpt_path)
        decoder.load_state_dict(decoder_ckpt["decoder"])
        self.decoder = decoder.cuda()
        self.decoder.eval()

        lr = hcfg("lr", float)
        wd = hcfg("wd", float)
        self.optimizer = AdamW(self.transfer.parameters(), lr, weight_decay=wd)

        self.epoch = 0
        self.global_step = 0
        self.best_chamfer = 1000.0

        self.ckpts_path = get_out_dir() / "ckpts"
        self.all_ckpts_path = get_out_dir() / "all_ckpts"

        if self.ckpts_path.exists():
            self.restore_from_last_ckpt()

        self.ckpts_path.mkdir(exist_ok=True)
        self.all_ckpts_path.mkdir(exist_ok=True)

    def logfn(self, values: Dict[str, Any]) -> None:
        wandb.log(values, step=self.global_step, commit=False)

    def train(self) -> None:
        num_epochs = hcfg("num_epochs", int)
        start_epoch = self.epoch

        for epoch in range(start_epoch, num_epochs):
            self.epoch = epoch

            self.transfer.train()

            desc = f"Epoch {epoch}/{num_epochs}"
            for batch in progress_bar(self.train_loader, desc=desc):
                _, _, embeddings_incomplete, embeddings_complete = batch
                embeddings_incomplete = embeddings_incomplete.cuda()
                embeddings_complete = embeddings_complete.cuda()

                embeddings_transfer = self.transfer(embeddings_incomplete)

                loss = F.mse_loss(embeddings_transfer, embeddings_complete)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if self.global_step % 10 == 0:
                    self.logfn({"train/loss": loss.item()})

                self.global_step += 1

            if epoch % 10 == 9 or epoch == num_epochs - 1:
                self.val("train")
                self.val("val")
                self.plot("train")
                self.plot("val")
                self.save_ckpt()

            if epoch % 50 == 0:
                self.save_ckpt(all=True)

    def val(self, split: str) -> None:
        loader = self.train_val_loader if split == "train" else self.val_loader
        self.transfer.eval()

        cdts = []
        fscores = []
        idx = 0

        for batch in progress_bar(loader, desc=f"Validating on {split} set"):
            incompletes, completes, embeddings_incomplete, embeddings_complete = batch
            bs = incompletes.shape[0]
            incompletes = incompletes.cuda()
            completes = completes.cuda()
            embeddings_incomplete = embeddings_incomplete.cuda()
            embeddings_complete = embeddings_complete.cuda()

            with torch.no_grad():
                embeddings_transfer = self.transfer(embeddings_incomplete)

            def udfs_func(coords: Tensor, indices: List[int]) -> Tensor:
                emb = embeddings_transfer[indices]
                pred = torch.sigmoid(self.decoder(emb, coords))
                pred = 1 - pred
                pred *= 0.1
                return pred

            pred_pcds = sample_pcds_from_udfs(udfs_func, bs, 4096, (-1, 1), 0.05, 0.02, 2048, 1)

            completes_2048 = random_point_sampling(completes, 2048)

            cd = chamfer_t(pred_pcds, completes_2048)
            cdts.extend([float(cd[i]) for i in range(bs)])

            f = f_score(pred_pcds, completes_2048, threshold=0.01)[0]
            fscores.extend([float(f[i]) for i in range(bs)])

            if idx > 9 and split == "train":
                break
            idx += 1

        mean_cdt = sum(cdts) / len(cdts)
        mean_fscore = sum(fscores) / len(fscores)

        self.logfn({f"{split}/cdt": mean_cdt})
        self.logfn({f"{split}/fscore": mean_fscore})

        if split == "val" and mean_cdt < self.best_chamfer:
            self.best_chamfer = mean_cdt
            self.save_ckpt(best=True)

    def plot(self, split: str) -> None:
        loader = self.train_val_loader if split == "train" else self.val_loader
        self.transfer.eval()

        loader_iter = iter(loader)
        for _ in range(randint(1, len(loader) - 1)):
            batch = next(loader_iter)

        incompletes, completes, embeddings_incomplete, embeddings_complete = batch
        bs = incompletes.shape[0]
        incompletes = incompletes.cuda()
        completes = completes.cuda()
        embeddings_incomplete = embeddings_incomplete.cuda()
        embeddings_complete = embeddings_complete.cuda()

        with torch.no_grad():
            embeddings_transfer = self.transfer(embeddings_incomplete)

        def udfs_func(coords: Tensor, indices: List[int]) -> Tensor:
            emb = embeddings_transfer[indices]
            pred = torch.sigmoid(self.decoder(emb, coords))
            pred = 1 - pred
            pred *= 0.1
            return pred

        pred_pcds = sample_pcds_from_udfs(udfs_func, bs, 4096, (-1, 1), 0.05, 0.02, 2048, 1)

        completes_2048 = random_point_sampling(completes, 2048)

        for i in range(bs):
            inc_wo3d = wandb.Object3D(incompletes[i].cpu().detach().numpy())
            compl_wo3d = wandb.Object3D(completes_2048[i].cpu().detach().numpy())
            pred_wo3d = wandb.Object3D(pred_pcds[i].cpu().detach().numpy())
            pcd_logs = {f"{split}/pcd_{i}": [inc_wo3d, compl_wo3d, pred_wo3d]}
            self.logfn(pcd_logs)

    def save_ckpt(self, best: bool = False, all: bool = False) -> None:
        ckpt = {
            "epoch": self.epoch,
            "best_chamfer": self.best_chamfer,
            "net": self.transfer.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

        if all:
            ckpt_path = self.all_ckpts_path / f"{self.epoch}.pt"
            torch.save(ckpt, ckpt_path)

        else:
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
            self.best_chamfer = ckpt["best_chamfer"]

            self.transfer.load_state_dict(ckpt["net"])
            self.optimizer.load_state_dict(ckpt["optimizer"])


run_cfg_file = sys.argv[1] if len(sys.argv) == 2 else None


@hmain(
    base_cfg_dir="cfg/bases",
    template_cfg_file="cfg/completion.yaml",
    run_cfg_file=run_cfg_file,
    parse_cmd_line=False,
    out_dir_root="../logs",
)
def main() -> None:
    wandb.init(
        entity="entity",
        project=f"inr2vec",
        name=get_run_name(),
        dir=str(get_out_dir()),
        config=get_cfg_copy(),
    )

    trainer = CompletionTrainer()
    trainer.train()


if __name__ == "__main__":
    main()
