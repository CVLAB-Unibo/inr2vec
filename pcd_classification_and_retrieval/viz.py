import sys

sys.path.append("..")

from pathlib import Path
from random import randint
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np
import torch
from hesiod import get_out_dir, hcfg, hmain
from pycarus.geometry.pcd import get_o3d_pcd_from_tensor, sample_pcds_from_udfs
from pycarus.learning.models.siren import SIREN
from torch import Tensor
from torch.utils.data import Dataset

from models.encoder import Encoder
from models.idecoder import ImplicitDecoder
from utils import get_mlp_params_as_matrix

import open3d as o3d  # isort: skip


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


@hmain(base_cfg_dir="cfg/bases", run_cfg_file=sys.argv[1], parse_cmd_line=False)
def main() -> None:
    ckpt_path = get_out_dir() / "ckpts/best.pt"
    ckpt = torch.load(ckpt_path)

    inrs_root = Path(hcfg("inrs_root", str))

    mlp_hdim = hcfg("mlp.hidden_dim", int)
    num_hidden_layers = hcfg("mlp.num_hidden_layers", int)
    mlp = SIREN(3, mlp_hdim, num_hidden_layers, 1)
    sample_sd = mlp.state_dict()
    dset = InrDataset(inrs_root, "test", sample_sd)

    encoder_cfg = hcfg("encoder", Dict[str, Any])
    encoder = Encoder(
        mlp_hdim,
        encoder_cfg["hidden_dims"],
        encoder_cfg["embedding_dim"],
    )
    encoder.load_state_dict(ckpt["encoder"])
    encoder = encoder.cuda()
    encoder.eval()

    decoder_cfg = hcfg("decoder", Dict[str, Any])
    decoder = ImplicitDecoder(
        encoder_cfg["embedding_dim"],
        decoder_cfg["input_dim"],
        decoder_cfg["hidden_dim"],
        decoder_cfg["num_hidden_layers_before_skip"],
        decoder_cfg["num_hidden_layers_after_skip"],
        decoder_cfg["out_dim"],
    )
    decoder.load_state_dict(ckpt["decoder"])
    decoder = decoder.cuda()
    decoder.eval()

    while True:
        idx = randint(0, len(dset) - 1)
        gt_pcd, matrix = dset[idx]
        matrix = matrix.cuda()

        with torch.no_grad():
            embeddings = encoder(matrix.unsqueeze(0))

        def udfs_func(coords: Tensor, indices: List[int]) -> Tensor:
            pred = torch.sigmoid(decoder(embeddings, coords))
            pred = 1 - pred
            pred *= 0.1
            return pred

        output_pcd = sample_pcds_from_udfs(udfs_func, 1, 4096, (-1, 1), 0.05, 0.02, 8192, 5)[0]

        gt_o3d = get_o3d_pcd_from_tensor(gt_pcd)
        gt_o3d.paint_uniform_color((0, 1, 0))
        output_o3d = get_o3d_pcd_from_tensor(output_pcd).translate((2, 0, 0))
        output_o3d.paint_uniform_color((0, 0, 1))

        o3d.visualization.draw_geometries([gt_o3d, output_o3d])


if __name__ == "__main__":
    main()
