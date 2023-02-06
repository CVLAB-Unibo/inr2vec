import sys

sys.path.append("..")

from math import ceil
from pathlib import Path
from typing import Callable, List

import h5py
import torch
import torch.nn.functional as F
from hesiod import hcfg, hmain
from pycarus.datasets.manifold40 import Manifold40
from pycarus.geometry.mesh import compute_sdf_from_mesh, get_o3d_mesh_from_tensors
from pycarus.geometry.pcd import random_point_sampling, shuffle_pcd
from pycarus.learning.models.siren import SIREN
from pycarus.transforms.pcd import NormalizePcdIntoUnitSphere, RandomScalePcd
from pycarus.utils import progress_bar
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from utils import get_mlps_batched_params, mlp_batched_forward


class InrsDatasetCreator:
    def __init__(self) -> None:
        self.dset_name = hcfg("dset_name", str)
        self.mesh_root = Path(hcfg("mesh_root", str))
        self.splits = hcfg("splits", List[str])

        self.num_queries_on_surface = hcfg("num_queries_on_surface", int)
        self.stds = hcfg("stds", List[float])
        self.num_points_per_std = hcfg("num_points_per_std", List[int])

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
        if self.dset_name == "manifold40":
            dset = Manifold40(self.mesh_root, split, vertices_transforms=transforms, only_version=0)
        else:
            raise ValueError("Unknown dataset.")
        return dset

    def create_dataset(self) -> None:
        for split in self.splits:
            global_idx = 0

            augs = [False]
            if "train" in split:
                augs += [True] * self.num_augmentations

            for num_aug, aug in enumerate(augs):
                if aug:
                    transforms = [
                        RandomScalePcd(2 / 3, 3 / 2),
                        NormalizePcdIntoUnitSphere(),
                    ]
                else:
                    transforms = [NormalizePcdIntoUnitSphere()]

                dset = self.get_dataset(split, transforms)

                loader = DataLoader(dset, self.num_parallel_mlps, num_workers=0)

                desc = f"Fitting {split} set (aug {num_aug + 1}/{len(augs)})"
                for batch in progress_bar(loader, desc, num_cols=80):
                    vertices, triangles, class_ids, num_vertices, num_triangles = batch
                    bs = len(vertices)

                    coords = []
                    labels = []

                    for idx in range(bs):
                        num_v = num_vertices[idx]
                        v = vertices[idx][:num_v]
                        num_t = num_triangles[idx]
                        t = triangles[idx][:num_t]
                        mesh_o3d = get_o3d_mesh_from_tensors(v, t)

                        mesh_coords, mesh_labels = compute_sdf_from_mesh(
                            mesh_o3d,
                            num_queries_on_surface=self.num_queries_on_surface,
                            queries_stds=self.stds,
                            num_queries_per_std=self.num_points_per_std,
                            coords_range=(-1, 1),
                        )
                        coords.append(mesh_coords)
                        labels.append(mesh_labels)

                    coords = torch.stack(coords, dim=0)
                    labels = torch.stack(labels, dim=0)

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

                        selected_labels = (selected_labels + 0.1) / 0.2
                        loss = F.binary_cross_entropy_with_logits(pred, selected_labels)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    for idx in range(bs):
                        v = vertices[idx]
                        num_v = num_vertices[idx]
                        t = triangles[idx]
                        num_t = num_triangles[idx]

                        class_id = class_ids[idx]

                        crd = coords[idx]
                        lbl = labels[idx]

                        flattened_params = [p[idx].view(-1) for p in batched_params]
                        flattened_params = torch.cat(flattened_params, dim=0)

                        h5_path = self.out_root / split / f"{global_idx}.h5"
                        h5_path.parent.mkdir(parents=True, exist_ok=True)

                        with h5py.File(h5_path, "w") as f:
                            f.create_dataset("vertices", data=v.detach().cpu().numpy())
                            f.create_dataset("num_vertices", data=num_v.detach().cpu().numpy())
                            f.create_dataset("triangles", data=t.detach().cpu().numpy())
                            f.create_dataset("num_triangles", data=num_t.detach().cpu().numpy())
                            f.create_dataset("params", data=flattened_params.detach().cpu().numpy())
                            f.create_dataset("class_id", data=class_id.detach().cpu().numpy())
                            if split == "train":
                                f.create_dataset("coords", data=crd.detach().cpu().numpy())
                                f.create_dataset("labels", data=lbl.detach().cpu().numpy())

                        global_idx += 1


@hmain(base_cfg_dir="cfg/bases", template_cfg_file="cfg/inrs_dataset.yaml", create_out_dir=False)
def main() -> None:
    dset_creator = InrsDatasetCreator()
    dset_creator.create_dataset()


if __name__ == "__main__":
    main()
