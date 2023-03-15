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
from torch import Tensor
from torch.utils.data import Dataset

from models.idecoder import ImplicitDecoder
from models.transfer import Transfer

import open3d as o3d  # isort: skip


class InrEmbeddingDataset(Dataset):
    def __init__(self, root: Path, split: str) -> None:
        super().__init__()

        self.root = root / split
        self.item_paths = sorted(self.root.glob("*.h5"), key=lambda x: int(x.stem))

    def __len__(self) -> int:
        return len(self.item_paths)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        with h5py.File(self.item_paths[index], "r") as f:
            embedding_incomplete = np.array(f.get("embedding_incomplete"))
            embedding_incomplete = torch.from_numpy(embedding_incomplete)
            embedding_complete = np.array(f.get("embedding_complete"))
            embedding_complete = torch.from_numpy(embedding_complete)

        return embedding_incomplete, embedding_complete


@hmain(
    base_cfg_dir="cfg/bases",
    run_cfg_file=sys.argv[1],
    parse_cmd_line=False,
    create_out_dir=False,
)
def main() -> None:
    dset_root = Path(hcfg("dset_root", str))
    dset = InrEmbeddingDataset(dset_root, "test")

    ckpt_path = get_out_dir() / "ckpts/best.pt"
    ckpt = torch.load(ckpt_path)

    embedding_dim = hcfg("embedding_dim", int)
    num_layers = hcfg("num_layers_transfer", int)
    transfer = Transfer(embedding_dim, num_layers)
    transfer.load_state_dict(ckpt["net"])
    transfer = transfer.cuda()
    transfer.eval()

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
    decoder = decoder.cuda()
    decoder.eval()

    while True:
        idx = randint(0, len(dset) - 1)
        print("Index:", idx)

        embedding_incomplete, embedding_complete = dset[idx]

        embedding_incomplete = embedding_incomplete.unsqueeze(0).cuda()
        embedding_complete = embedding_complete.unsqueeze(0).cuda()

        def udfs_func_i(coords: Tensor, indices: List[int]) -> Tensor:
            pred = torch.sigmoid(decoder(embedding_incomplete, coords))
            pred = 1 - pred
            pred *= 0.1
            return pred

        pred_inc = sample_pcds_from_udfs(udfs_func_i, 1, 4096, (-1, 1), 0.05, 0.02, 8192, 3)[0]

        with torch.no_grad():
            emb_tr = transfer(embedding_incomplete)

        def udfs_func_t(coords: Tensor, indices: List[int]) -> Tensor:
            pred = torch.sigmoid(decoder(emb_tr, coords))
            pred = 1 - pred
            pred *= 0.1
            return pred

        pred_tr = sample_pcds_from_udfs(udfs_func_t, 1, 4096, (-1, 1), 0.05, 0.02, 8192, 3)[0]

        def udfs_func_c(coords: Tensor, indices: List[int]) -> Tensor:
            pred = torch.sigmoid(decoder(embedding_complete, coords))
            pred = 1 - pred
            pred *= 0.1
            return pred

        pred_compl = sample_pcds_from_udfs(udfs_func_c, 1, 4096, (-1, 1), 0.05, 0.02, 8192, 3)[0]

        pred_inc_o3d = get_o3d_pcd_from_tensor(pred_inc)
        pred_tr_o3d = get_o3d_pcd_from_tensor(pred_tr).translate((2, 0, 0))
        pred_compl_o3d = get_o3d_pcd_from_tensor(pred_compl).translate((4, 0, 0))

        o3d.visualization.draw_geometries(
            [
                pred_inc_o3d,
                pred_tr_o3d,
                pred_compl_o3d,
            ]
        )


if __name__ == "__main__":
    main()
