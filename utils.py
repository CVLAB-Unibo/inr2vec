import collections
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from pycarus.learning.models.siren import SIREN
from torch import Tensor


def get_mlps_batched_params(mlps: List[SIREN]) -> List[Tensor]:
    params = []
    for i in range(len(mlps)):
        params.append(list(mlps[i].parameters()))

    batched_params = []
    for i in range(len(params[0])):
        p = torch.stack([p[i] for p in params], dim=0)
        p = torch.clone(p.detach())
        p.requires_grad = True
        batched_params.append(p)

    return batched_params


def flatten_mlp_params(sd: Dict[str, Any]) -> Tensor:
    all_params = []
    for k in sd:
        all_params.append(sd[k].view(-1))
    all_params = torch.cat(all_params, dim=-1)
    return all_params


def unflatten_mlp_params(
    params: Tensor,
    sample_sd: Dict[str, Any],
) -> Dict[str, Any]:
    sd = collections.OrderedDict()

    start = 0
    for k in sample_sd:
        end = start + sample_sd[k].numel()
        layer_params = params[start:end].view(sample_sd[k].shape)
        sd[k] = layer_params
        start = end

    return sd


def get_mlp_params_as_matrix(flattened_params: Tensor, sd: Dict[str, Any]) -> Tensor:
    params_shapes = [p.shape for p in sd.values()]
    feat_dim = params_shapes[0][0]
    start = params_shapes[0].numel() + params_shapes[1].numel()
    end = params_shapes[-1].numel() + params_shapes[-2].numel()
    params = flattened_params[start:-end]
    return params.reshape((-1, feat_dim))


def mlp_batched_forward(batched_params: List[Tensor], coords: Tensor) -> Tensor:
    num_layers = len(batched_params) // 2

    f = coords

    for i in range(num_layers):
        weights = batched_params[i * 2]
        biases = batched_params[i * 2 + 1]

        f = torch.bmm(f, weights.permute(0, 2, 1)) + biases.unsqueeze(1)

        if i < num_layers - 1:
            f = torch.sin(30 * f)

    return f.squeeze(-1)


def focal_loss(pred: Tensor, gt: Tensor, alpha: float = 0.1, gamma: float = 3) -> Tensor:
    alpha_w = torch.tensor([alpha, 1 - alpha]).cuda()

    bce_loss = F.binary_cross_entropy_with_logits(pred, gt.float(), reduction="none")
    bce_loss = bce_loss.view(-1)

    gt = gt.type(torch.long)
    at = alpha_w.gather(0, gt.view(-1))
    pt = torch.exp(-bce_loss)
    f_loss = at * ((1 - pt) ** gamma) * bce_loss

    return f_loss.mean()


def get_class_to_parts(dset_name: str) -> Dict[str, List[int]]:
    shapenet_partseg = {
        "02691156": [0, 1, 2, 3],
        "02773838": [4, 5],
        "02954340": [6, 7],
        "02958343": [8, 9, 10, 11],
        "03001627": [12, 13, 14, 15],
        "03261776": [16, 17, 18],
        "03467517": [19, 20, 21],
        "03624134": [22, 23],
        "03636649": [24, 25, 26, 27],
        "03642806": [28, 29],
        "03790512": [30, 31, 32, 33, 34, 35],
        "03797390": [36, 37],
        "03948459": [38, 39, 40],
        "04099429": [41, 42, 43],
        "04225987": [44, 45, 46],
        "04379243": [47, 48, 49],
    }

    gallery = {"shapenet-partseg": shapenet_partseg}

    return gallery[dset_name]
