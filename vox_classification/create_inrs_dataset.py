import sys

sys.path.append("..")

from math import ceil
from pathlib import Path
from typing import Callable, List

import h5py
import torch
from einops import rearrange, repeat
from hesiod import hcfg, hmain
from pycarus.datasets.ply import PlyDataset
from pycarus.geometry.pcd import farthest_point_sampling, random_point_sampling, shuffle_pcd
from pycarus.geometry.pcd import voxelize_pcd
from pycarus.learning.models.siren import SIREN
from pycarus.transforms.pcd import JitterPcd, NormalizePcdIntoUnitSphere, RandomScalePcd
from pycarus.utils import progress_bar
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from utils import focal_loss, get_mlps_batched_params, mlp_batched_forward


class InrsDatasetCreator:
    def __init__(self) -> None:
        self.dset_name = hcfg("dset_name", str)
        self.pcd_root = Path(hcfg("pcd_root", str))
        self.splits = hcfg("splits", List[str])
        self.num_points_pcd = hcfg("num_points_pcd", int)
        self.vox_res = hcfg("vox_res", int)

        self.num_required_train_shapes = hcfg("num_required_train_shapes", int)
        dset = self.get_dataset("train")
        num_train_shapes = len(dset)  # type: ignore
        self.num_augmentations = ceil(self.num_required_train_shapes / num_train_shapes) - 1

        self.num_points_fitting = hcfg("num_points_fitting", int)
        self.num_parallel_mlps = hcfg("num_parallel_mlps", int)
        self.hdim = hcfg("mlp.hidden_dim", int)
        self.num_hidden_layers = hcfg("mlp.num_hidden_layers", int)
        self.mlp_init_path = Path(hcfg("mlp.init_path", str))

        self.num_steps = hcfg("num_steps", int)
        self.lr = hcfg("lr", float)

        self.out_root = Path(hcfg("out_root", str))
        self.out_root.mkdir(parents=True)

    def build_mlp(self) -> SIREN:
        mlp = SIREN(
            input_dim=3,
            hidden_dim=self.hdim,
            num_hidden_layers=self.num_hidden_layers,
            out_dim=1,
        )

        mlp.load_state_dict(torch.load(self.mlp_init_path))

        return mlp

    def get_dataset(self, split: str, transforms: List[Callable] = []) -> Dataset:
        if self.dset_name == "shapenet10":
            dset = PlyDataset(self.pcd_root, split, transforms)
        else:
            raise ValueError("Unknown dataset.")
        return dset

    def create_dataset(self) -> None:
        for split in self.splits:
            global_idx = 0

            augs = [False]
            if "train" in split:
                augs += [True] * (self.num_augmentations + 1)

            for aug_idx, aug in enumerate(augs):
                if aug:
                    transforms = [
                        RandomScalePcd(2 / 3, 3 / 2),
                        JitterPcd(sigma=0.01, clip=0.05),
                        NormalizePcdIntoUnitSphere(),
                    ]
                else:
                    transforms = [NormalizePcdIntoUnitSphere()]

                dset = self.get_dataset(split, transforms)

                loader = DataLoader(
                    dset,
                    batch_size=self.num_parallel_mlps,
                    shuffle=False,
                    num_workers=0,
                )

                desc = f"Fitting {split} set ({aug_idx + 1}/{len(augs)})"
                for batch in progress_bar(loader, desc, 80):
                    pcds, class_ids = batch
                    bs = pcds.shape[0]

                    if pcds.shape[1] != self.num_points_pcd:
                        pcds = farthest_point_sampling(pcds, self.num_points_pcd)

                    vgrids, centroids = voxelize_pcd(pcds, self.vox_res, -1, 1)

                    coords = repeat(centroids, "r1 r2 r3 d -> b r1 r2 r3 d", b=bs)
                    coords = rearrange(coords, "b r1 r2 r3 d -> b (r1 r2 r3) d")
                    labels = rearrange(vgrids, "b r1 r2 r3 -> b (r1 r2 r3)")

                    coords_and_labels = torch.cat((coords, labels.unsqueeze(-1)), dim=-1).cuda()
                    coords_and_labels = shuffle_pcd(coords_and_labels)

                    mlps = [self.build_mlp().cuda() for _ in range(bs)]
                    batched_params = get_mlps_batched_params(mlps)

                    optimizer = Adam(batched_params, lr=self.lr)

                    for _ in progress_bar(range(self.num_steps)):
                        selected_c_and_l = random_point_sampling(
                            coords_and_labels,
                            self.num_points_fitting,
                        )

                        selected_coords = selected_c_and_l[:, :, :3]
                        selected_labels = selected_c_and_l[:, :, 3]

                        pred = mlp_batched_forward(batched_params, selected_coords)
                        loss = focal_loss(pred, selected_labels)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    for idx in range(bs):
                        pcd = pcds[idx]
                        class_id = class_ids[idx]

                        flattened_params = [p[idx].view(-1) for p in batched_params]
                        flattened_params = torch.cat(flattened_params, dim=0)

                        h5_path = self.out_root / split / f"{global_idx}.h5"
                        h5_path.parent.mkdir(parents=True, exist_ok=True)

                        with h5py.File(h5_path, "w") as f:
                            f.create_dataset("pcd", data=pcd.detach().cpu().numpy())
                            f.create_dataset("params", data=flattened_params.detach().cpu().numpy())
                            f.create_dataset("class_id", data=class_id.detach().cpu().numpy())

                        global_idx += 1


@hmain(base_cfg_dir="cfg/bases", template_cfg_file="cfg/inrs_dataset.yaml", create_out_dir=False)
def main() -> None:
    dset_creator = InrsDatasetCreator()
    dset_creator.create_dataset()


if __name__ == "__main__":
    main()
