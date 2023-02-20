import sys

sys.path.append("..")

import logging
import os
from pathlib import Path
from random import randint
from typing import Any, Dict, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from hesiod import get_cfg_copy, get_out_dir, get_run_name, hcfg, hmain
from pycarus.geometry.mesh import marching_cubes
from pycarus.geometry.pcd import random_point_sampling
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

T_ITEM = Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]


class InrDataset(Dataset):
    def __init__(self, inrs_root: Path, split: str, sample_sd: Dict[str, Any]) -> None:
        super().__init__()

        self.inrs_root = inrs_root / split
        self.mlps_paths = sorted(self.inrs_root.glob("*.h5"), key=lambda x: int(x.stem))
        self.sample_sd = sample_sd
        self.split = split

    def __len__(self) -> int:
        return len(self.mlps_paths)

    def __getitem__(self, index: int) -> T_ITEM:
        with h5py.File(self.mlps_paths[index], "r") as f:
            vertices = torch.from_numpy(np.array(f.get("vertices")))
            num_vertices = torch.from_numpy(np.array(f.get("num_vertices")))
            triangles = torch.from_numpy(np.array(f.get("triangles")))
            num_triangles = torch.from_numpy(np.array(f.get("num_triangles")))
            params = torch.from_numpy(np.array(f.get("params"))).float()
            matrix = get_mlp_params_as_matrix(params, self.sample_sd)

            if self.split == "train":
                coords = torch.from_numpy(np.array(f.get("coords")))
                labels = torch.from_numpy(np.array(f.get("labels")))
            else:
                coords = torch.zeros(0, 3)
                labels = torch.zeros(0)

        return vertices, num_vertices, triangles, num_triangles, coords, labels, matrix


class Inr2vecTrainer:
    def __init__(self) -> None:
        inrs_root = Path(hcfg("inrs_root", str))
        self.num_points_fitting = hcfg("num_points_fitting", int)

        mlp_hdim = hcfg("mlp.hidden_dim", int)
        num_hidden_layers = hcfg("mlp.num_hidden_layers", int)
        mlp = SIREN(3, mlp_hdim, num_hidden_layers, 1)
        sample_sd = mlp.state_dict()

        train_split = hcfg("train_split", str)
        train_dset = InrDataset(inrs_root, train_split, sample_sd)
        train_bs = hcfg("train_bs", int)
        self.train_loader = DataLoader(train_dset, batch_size=train_bs, num_workers=8, shuffle=True)

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
                _, _, _, _, coords, labels, matrices = batch
                coords = coords.cuda()
                labels = labels.cuda()
                matrices = matrices.cuda()

                coords_and_labels = torch.cat((coords, labels.unsqueeze(-1)), dim=-1)
                selected_c_and_l = random_point_sampling(coords_and_labels, self.num_points_fitting)

                selected_coords = selected_c_and_l[:, :, :3]
                selected_labels = selected_c_and_l[:, :, 3]

                embeddings = self.encoder(matrices)
                pred = self.decoder(embeddings, selected_coords)

                selected_labels = (selected_labels + 0.1) / 0.2
                loss = F.binary_cross_entropy_with_logits(pred, selected_labels)

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

    @torch.no_grad()
    def val(self, split: str) -> None:
        loader = self.train_loader if split == "train" else self.val_loader

        self.encoder.eval()
        self.decoder.eval()

        cdts = []
        fscores = []
        idx = 0

        for batch in progress_bar(loader, desc=f"Validating on {split} set"):
            vertices, num_vertices, _, _, _, _, matrices = batch
            matrices = matrices.cuda()
            bs = len(vertices)

            embeddings = self.encoder(matrices)

            for i in range(bs):
                gt_vertices = vertices[i][: num_vertices[i]]

                try:

                    def levelset_func(c: Tensor) -> Tensor:
                        pred = self.decoder(embeddings[i].unsqueeze(0), c.unsqueeze(0))
                        pred = torch.sigmoid(pred.squeeze(0))
                        pred *= 0.2
                        pred -= 0.1
                        return pred

                    pred_v, _ = marching_cubes(levelset_func, (-1, 1), 64, level=0.01)

                    pred_v = pred_v[:, :3].float().cuda()
                    pred_v = random_point_sampling(pred_v, gt_vertices.shape[0])

                    cdts.append(float(chamfer_t(pred_v, gt_vertices.cuda())))
                    fscores.append(float(f_score(pred_v, gt_vertices.cuda(), threshold=0.01)[0]))
                except:
                    cdts.append(100.0)
                    fscores.append(0.0)

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

    @torch.no_grad()
    def plot(self, split: str) -> None:
        loader = self.train_loader if split == "train" else self.val_loader_shuffled

        self.encoder.eval()
        self.decoder.eval()

        loader_iter = iter(loader)
        batch = next(loader_iter)

        gt_v, num_v, _, _, _, _, matrices = batch
        matrices = matrices.cuda()
        bs = len(gt_v)

        embeddings = self.encoder(matrices)

        for i in range(bs):
            gt_vert = gt_v[i][: num_v[i]]

            try:
                gt_pcd = random_point_sampling(gt_vert[:, :3].cuda(), 2048)
                gt_wo3d = wandb.Object3D(gt_pcd.cpu().detach().numpy())

                def levelset_func(c: Tensor) -> Tensor:
                    pred = self.decoder(embeddings[i].unsqueeze(0), c.unsqueeze(0))
                    pred = torch.sigmoid(pred.squeeze(0))
                    pred *= 0.2
                    pred -= 0.1
                    return pred

                pred_v, _ = marching_cubes(levelset_func, (-1, 1), 64, level=0.01)

                pred_v = pred_v[:, :3].float().cuda()
                pred_pcd = random_point_sampling(pred_v, 2048)
                pred_wo3d = wandb.Object3D(pred_pcd.cpu().detach().numpy())

                mesh_logs = {f"{split}/mesh_{i}": [gt_wo3d, pred_wo3d]}
                self.logfn(mesh_logs)
            except:
                pass

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
