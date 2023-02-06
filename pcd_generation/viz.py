import sys

sys.path.append("..")

from pathlib import Path
from random import randint
from typing import Any, Dict, List

import numpy as np
import torch
from hesiod import get_out_dir, hcfg, hmain
from pycarus.geometry.pcd import get_o3d_pcd_from_tensor, sample_pcds_from_udfs
from torch import Tensor

from models.idecoder import ImplicitDecoder

import open3d as o3d  # isort: skip

latent_gan_embeddings_path = Path("/path/to/latent/gan/embeddings")


@hmain(base_cfg_dir="cfg/bases", run_cfg_file=sys.argv[1], parse_cmd_line=False)
def main() -> None:
    ckpt_path = get_out_dir() / "ckpts/best.pt"
    ckpt = torch.load(ckpt_path)

    embeddings = np.load(latent_gan_embeddings_path)["embeddings"]
    embeddings = torch.from_numpy(embeddings)

    encoder_cfg = hcfg("encoder", Dict[str, Any])
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
        idx = randint(0, embeddings.shape[0])
        emb = embeddings[idx].unsqueeze(0).cuda()

        def udfs_func(coords: Tensor, indices: List[int]) -> Tensor:
            pred = torch.sigmoid(decoder(emb, coords))
            pred = 1 - pred
            pred *= 0.1
            return pred

        output_pcd = sample_pcds_from_udfs(udfs_func, 1, 4096, (-1, 1), 0.05, 0.02, 8192, 5)[0]
        output_o3d = get_o3d_pcd_from_tensor(output_pcd)

        o3d.visualization.draw_geometries([output_o3d])


if __name__ == "__main__":
    main()
