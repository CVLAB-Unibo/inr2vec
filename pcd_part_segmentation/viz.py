import sys

sys.path.append("..")

from pathlib import Path
from random import randint
from typing import Any, Dict, Tuple

import h5py
import numpy as np
import torch
from hesiod import get_out_dir, hcfg, hmain
from pycarus.datasets.shapenet_part import ShapeNetPartSegmentation
from torch import Tensor
from torch.utils.data import Dataset

from models.idecoder import ImplicitDecoder

import open3d as o3d  # isort: skip


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


def get_one_hot_encoding(x: Tensor, num_classes: int) -> Tensor:
    one_hot = torch.eye(num_classes)[x.cpu()]
    one_hot = one_hot.to(x.device)
    return one_hot


@hmain(base_cfg_dir="cfg/bases", run_cfg_file=sys.argv[1], parse_cmd_line=False)
def main() -> None:
    ckpt_path = get_out_dir() / "ckpts/best.pt"
    ckpt = torch.load(ckpt_path)

    embeddings_root = Path(hcfg("dset_root", str))
    dset = InrEmbeddingDataset(embeddings_root, "test")
    num_classes = hcfg("num_classes", int)
    num_part = hcfg("num_part", int)

    encoder_cfg = hcfg("encoder", Dict[str, Any])
    decoder_cfg = hcfg("decoder", Dict[str, Any])
    decoder = ImplicitDecoder(
        encoder_cfg["embedding_dim"] + num_classes,
        decoder_cfg["input_dim"],
        decoder_cfg["hidden_dim"],
        decoder_cfg["num_hidden_layers_before_skip"],
        decoder_cfg["num_hidden_layers_after_skip"],
        num_part,
    )
    decoder.load_state_dict(ckpt["decoder"])
    decoder = decoder.cuda()
    decoder.eval()

    while True:
        idx = randint(0, len(dset) - 1)
        gt_pcd, embedding, class_id, gt_part_labels = dset[idx]
        gt_pcd = gt_pcd.cuda()
        embedding = embedding.cuda()

        gt_pcd_o3d = ShapeNetPartSegmentation.color_pcd(gt_pcd, gt_part_labels)

        class_onehots = get_one_hot_encoding(class_id.unsqueeze(0), num_classes)
        class_onehots = class_onehots.cuda()
        embeddings = torch.cat([embedding.unsqueeze(0), class_onehots], dim=1)
        pred_logits = decoder(embeddings, gt_pcd.unsqueeze(0)).squeeze(0)
        pred_part_labels = torch.argmax(pred_logits, dim=-1)

        pred_pcd_o3d = ShapeNetPartSegmentation.color_pcd(gt_pcd, pred_part_labels)
        pred_pcd_o3d = pred_pcd_o3d.translate((2, 0, 0))

        o3d.visualization.draw_geometries([gt_pcd_o3d, pred_pcd_o3d])


if __name__ == "__main__":
    main()
