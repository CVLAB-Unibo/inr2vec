import sys

sys.path.append("..")

from pathlib import Path
from typing import List

import h5py
import torch
import torch.nn.functional as F
from hesiod import hcfg, hmain
from pycarus.datasets.mvp import Mvp
from pycarus.geometry.pcd import compute_udf_from_pcd, random_point_sampling, shuffle_pcd
from pycarus.learning.models.siren import SIREN
from pycarus.transforms.pcd import NormalizePcdIntoUnitSphere
from pycarus.utils import progress_bar
from torch.optim import Adam
from torch.utils.data import DataLoader

from utils import get_mlps_batched_params, mlp_batched_forward


class InrsDatasetCreator:
    def __init__(self) -> None:
        self.dset_name = hcfg("dset_name", str)
        self.pcd_root = Path(hcfg("pcd_root", str))
        self.splits = hcfg("splits", List[str])
        self.num_points_pcd = hcfg("num_points_pcd", int)

        self.num_queries_on_surface = hcfg("num_queries_on_surface", int)
        self.stds = hcfg("stds", List[float])
        self.num_points_per_std = hcfg("num_points_per_std", List[int])

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

    def create_dataset(self) -> None:
        for split in self.splits:
            global_idx = 0

            dset = Mvp(
                self.pcd_root,
                split,
                self.num_points_pcd,
                transforms_all=[NormalizePcdIntoUnitSphere()],
            )

            loader = DataLoader(
                dset,
                batch_size=self.num_parallel_mlps // 2,
                shuffle=False,
                num_workers=8,
            )

            for batch in progress_bar(loader, f"Fitting {split} set", 80):
                categories, names, incompletes, completes = batch

                pcds = torch.cat([incompletes, completes], dim=0)
                bs = pcds.shape[0]
                pcds = pcds.cuda()

                coords = []
                labels = []
                for idx in range(bs):
                    pcd_coords, pcd_labels = compute_udf_from_pcd(
                        pcds[idx],
                        self.num_queries_on_surface,
                        self.stds,
                        self.num_points_per_std,
                        coords_range=(-1, 1),
                        convert_to_bce_labels=True,
                    )
                    coords.append(pcd_coords)
                    labels.append(pcd_labels)

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
                    loss = F.binary_cross_entropy_with_logits(pred, selected_labels)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                for idx in range(bs // 2):
                    incomplete = pcds[idx]
                    complete = pcds[(bs // 2) + idx]

                    category = categories[idx]
                    name = names[idx]

                    batched_params_inc = batched_params[idx]
                    flattened_params_inc = [p[idx].view(-1) for p in batched_params_inc]
                    flattened_params_inc = torch.cat(flattened_params_inc, dim=0)

                    batched_params_compl = batched_params[(bs // 2) + idx]
                    flattened_params_compl = [p[idx].view(-1) for p in batched_params_compl]
                    flattened_params_compl = torch.cat(flattened_params_compl, dim=0)

                    h5_path = self.out_root / split / f"{global_idx}.h5"
                    h5_path.parent.mkdir(parents=True, exist_ok=True)

                    with h5py.File(h5_path, "w") as f:
                        f.create_dataset("incomplete", data=incomplete.detach().cpu().numpy())
                        f.create_dataset("complete", data=complete.detach().cpu().numpy())
                        params_incomplete = flattened_params_inc.detach().cpu().numpy()
                        f.create_dataset("params_incomplete", data=params_incomplete)
                        params_complete = flattened_params_compl.detach().cpu().numpy()
                        f.create_dataset("params_complete", data=params_complete)
                        f.create_dataset("category", data=category)
                        f.create_dataset("name", data=name)

                    global_idx += 1


@hmain(base_cfg_dir="cfg/bases", template_cfg_file="cfg/inrs_dataset.yaml", create_out_dir=False)
def main() -> None:
    dset_creator = InrsDatasetCreator()
    dset_creator.create_dataset()


if __name__ == "__main__":
    main()
