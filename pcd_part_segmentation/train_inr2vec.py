import sys

sys.path.append("..")

import logging
import os
from pathlib import Path
from random import randint
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from hesiod import get_cfg_copy, get_out_dir, get_run_name, hcfg, hmain
from pycarus.geometry.pcd import compute_udf_from_pcd, sample_pcds_from_udfs
from pycarus.learning.models.siren import SIREN
from pycarus.metrics.chamfer_distance import chamfer_t
from pycarus.metrics.f_score import f_score
from pycarus.utils import progress_bar
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from models.encoder import Encoder
from models.idecoder import ImplicitDecoder
from utils import get_mlp_params_as_matrix

logging.disable(logging.INFO)
os.environ["WANDB_SILENT"] = "true"


class InrDataset(Dataset):
    def __init__(self, inrs_root: Path, split: str, sample_sd: Dict[str, Any]) -> None:
        super().__init__()

        self.inrs_root = inrs_root / split
        self.mlps_paths = sorted(self.inrs_root.glob("*.h5"), key=lambda x: int(x.stem))
        self.sample_sd = sample_sd

    def __len__(self) -> int:
        return len(self.mlps_paths)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        with h5py.File(self.mlps_paths[index], "r") as f:
            pcd = torch.from_numpy(np.array(f.get("pcd")))
            params = np.array(f.get("params"))
            params = torch.from_numpy(params).float()
            matrix = get_mlp_params_as_matrix(params, self.sample_sd)

        return pcd, matrix


class Inr2vecTrainer:
    def __init__(self) -> None:
        inrs_root = Path(hcfg("inrs_root", str))

        self.num_queries_on_surface = hcfg("num_queries_on_surface", int)
        self.stds = hcfg("stds", List[float])
        self.num_points_per_std = hcfg("num_points_per_std", List[int])

        mlp_hdim = hcfg("mlp.hidden_dim", int)
        num_hidden_layers = hcfg("mlp.num_hidden_layers", int)
        mlp = SIREN(3, mlp_hdim, num_hidden_layers, 1)
        sample_sd = mlp.state_dict()

        train_split = hcfg("train_split", str)
        train_dset = InrDataset(inrs_root, train_split, sample_sd)
        train_bs = hcfg("train_bs", int)
        self.train_loader = DataLoader(
            train_dset,
            batch_size=train_bs,
            num_workers=8,
            shuffle=True,
        )

        val_split = hcfg("val_split", str)
        val_dset = InrDataset(inrs_root, val_split, sample_sd)
        val_bs = hcfg("val_bs", int)
        self.val_loader = DataLoader(val_dset, batch_size=val_bs, num_workers=8)
        self.val_loader_shuffled = DataLoader(
            val_dset,
            batch_size=val_bs,
            num_workers=8,
            shuffle=True,
        )

        encoder_cfg = hcfg("encoder", Dict[str, Any])
        encoder = Encoder(
            mlp_hdim,
            encoder_cfg["hidden_dims"],
            encoder_cfg["embedding_dim"],
        )
        self.encoder = encoder.cuda()

        decoder_cfg = hcfg("decoder", Dict[str, Any])
        decoder = ImplicitDecoder(
            encoder_cfg["embedding_dim"],
            decoder_cfg["input_dim"],
            decoder_cfg["hidden_dim"],
            decoder_cfg["num_hidden_layers_before_skip"],
            decoder_cfg["num_hidden_layers_after_skip"],
            decoder_cfg["out_dim"],
        )
        self.decoder = decoder.cuda()

        lr = hcfg("lr", float)
        wd = hcfg("wd", float)
        params = list(self.encoder.parameters())
        params.extend(self.decoder.parameters())
        self.optimizer = AdamW(params, lr, weight_decay=wd)

        self.epoch = 0
        self.global_step = 0
        self.best_chamfer = float("inf")

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

            self.encoder.train()
            self.decoder.train()

            desc = f"Epoch {epoch}/{num_epochs}"
            for batch in progress_bar(self.train_loader, desc=desc):
                pcds, matrices = batch
                matrices = matrices.cuda()
                pcds = pcds.cuda()

                coords = []
                labels = []
                for i in range(pcds.shape[0]):
                    crd, lbl = compute_udf_from_pcd(
                        pcds[i],
                        self.num_queries_on_surface,
                        self.stds,
                        self.num_points_per_std,
                        (-1, 1),
                        convert_to_bce_labels=True,
                    )
                    coords.append(crd)
                    labels.append(lbl)
                coords = torch.stack(coords, dim=0).cuda()
                labels = torch.stack(labels, dim=0).cuda()

                embeddings = self.encoder(matrices)

                pred = self.decoder(embeddings, coords)

                loss = F.binary_cross_entropy_with_logits(pred, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if self.global_step % 10 == 0:
                    self.logfn({"train/loss": loss.item()})

                self.global_step += 1

            if epoch % 10 == 0 or epoch == num_epochs - 1:
                self.val("train")
                self.val("val")
                self.plot("train")
                self.plot("val")

            if epoch % 50 == 0:
                self.save_ckpt(all=True)

            self.save_ckpt()

    def val(self, split: str) -> None:
        loader = self.train_loader if split == "train" else self.val_loader

        self.encoder.eval()
        self.decoder.eval()

        cdts = []
        fscores = []
        idx = 0
        for batch in progress_bar(loader, desc=f"Validating on {split} set"):
            pcds, matrices = batch
            pcds = pcds.cuda()
            matrices = matrices.cuda()

            bs = pcds.shape[0]

            with torch.no_grad():
                embeddings = self.encoder(matrices)

            pred_pcds = []

            for i in range(bs):
                emb = embeddings[i].unsqueeze(0)

                def udfs_func(coords: Tensor, indices: List[int]) -> Tensor:
                    pred = torch.sigmoid(self.decoder(emb, coords))
                    pred = 1 - pred
                    pred *= 0.1
                    return pred

                pred_pcd = sample_pcds_from_udfs(udfs_func, 1, 4096, (-1, 1), 0.05, 0.02, 2048, 1)
                pred_pcds.append(pred_pcd[0])

            pred_pcds = torch.stack(pred_pcds, dim=0)

            cd = chamfer_t(pred_pcds, pcds)
            cdts.extend([float(cd[i]) for i in range(bs)])

            f = f_score(pred_pcds, pcds, threshold=0.01)[0]
            fscores.extend([float(f[i]) for i in range(bs)])

            if idx > 99 and split == "train":
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
        loader = self.train_loader if split == "train" else self.val_loader_shuffled

        self.encoder.eval()
        self.decoder.eval()

        loader_iter = iter(loader)
        batch = next(loader_iter)

        pcds, matrices = batch
        pcds = pcds.cuda()
        matrices = matrices.cuda()

        bs = pcds.shape[0]

        with torch.no_grad():
            embeddings = self.encoder(matrices)

        pred_pcds = []

        for i in range(bs):
            emb = embeddings[i].unsqueeze(0)

            def udfs_func(coords: Tensor, indices: List[int]) -> Tensor:
                pred = torch.sigmoid(self.decoder(emb, coords))
                pred = 1 - pred
                pred *= 0.1
                return pred

            pred_pcd = sample_pcds_from_udfs(udfs_func, 1, 4096, (-1, 1), 0.05, 0.02, 2048, 1)
            pred_pcds.append(pred_pcd[0])

        for i in range(bs):
            gt_wo3d = wandb.Object3D(pcds[i].cpu().detach().numpy())
            pred_wo3d = wandb.Object3D(pred_pcds[i].cpu().detach().numpy())
            pcd_logs = {f"{split}/pcd_{i}": [gt_wo3d, pred_wo3d]}
            self.logfn(pcd_logs)

    def save_ckpt(self, best: bool = False, all: bool = False) -> None:
        ckpt = {
            "epoch": self.epoch,
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_chamfer": self.best_chamfer,
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

            self.encoder.load_state_dict(ckpt["encoder"])
            self.decoder.load_state_dict(ckpt["decoder"])
            self.optimizer.load_state_dict(ckpt["optimizer"])


run_cfg_file = sys.argv[1] if len(sys.argv) == 2 else None


@hmain(
    base_cfg_dir="cfg/bases",
    template_cfg_file="cfg/inr2vec.yaml",
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

    trainer = Inr2vecTrainer()
    trainer.train()


if __name__ == "__main__":
    main()
