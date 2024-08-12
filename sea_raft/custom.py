from dataclasses import dataclass
from typing import Any, List
import torch
import torch.nn.functional as F

from .core.raft import RAFT


def get_model_config():
    return {
        "name": "spring-M",
        "dataset": "spring",
        "gpus": [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7
        ],
        "use_var": True,
        "var_min": 0,
        "var_max": 10,
        "pretrain": "resnet34",
        "initial_dim": 64,
        "block_dims": [
            64,
            128,
            256
        ],
        "radius": 4,
        "dim": 128,
        "num_blocks": 2,
        "iters": 4,
        "image_size": [
            540,
            960
        ],
        "scale": -1,
        "batch_size": 32,
        "epsilon": 1e-8,
        "lr": 4e-4,
        "wdecay": 1e-5,
        "dropout": 0,
        "clip": 1.0,
        "gamma": 0.85,
        "num_steps": 120000,
        "restore_ckpt": None,
        "coarse_config": None
    }


@dataclass
class ModelArgs:
    name: str = None
    dataset: str = None
    gpus: List[int] = None
    use_var: bool = None
    var_min: int = None
    var_max: int = None
    pretrain: str = None
    initial_dim: int = None
    block_dims: List[int] = None
    radius: int = None
    dim: int = None
    num_blocks: int  = None
    iters: int = None
    image_size: List[int] = None
    scale: int = None
    batch_size: int = None
    epsilon: float = None
    lr: float = None
    wdecay: float = None
    dropout: float = None
    clip: float = None
    gamma: float = None
    num_steps: int = None
    restore_ckpt: Any = None
    coarse_config: Any = None


def forward_flow(args, model, image1, image2):
    output = model(image1, image2, iters=args.iters, test_mode=True)
    flow_final = output['flow'][-1]
    info_final = output['info'][-1]
    return flow_final, info_final


@torch.no_grad
def calc_flow(args, model, image1, image2):
    img1 = F.interpolate(image1, scale_factor=2 ** args.scale, mode='bilinear', align_corners=False)
    img2 = F.interpolate(image2, scale_factor=2 ** args.scale, mode='bilinear', align_corners=False)
    H, W = img1.shape[2:]
    flow, info = forward_flow(args, model, img1, img2)
    flow_down = F.interpolate(flow, scale_factor=0.5 ** args.scale, mode='bilinear', align_corners=False) * (0.5 ** args.scale)
    info_down = F.interpolate(info, scale_factor=0.5 ** args.scale, mode='area')
    return flow_down, info_down


class RaftWrapper(torch.nn.Module):
    def __init__(self, model, model_args):
        super(RaftWrapper, self).__init__()
        self.model = model
        self.args = model_args

    def forward(self, image1, image2):
        flow_down, info_down = calc_flow(self.args, self.model, image1, image2)
        return flow_down, info_down
    
    def eval(self):
        self.model.eval()


def get_model(checkpoint_path):
    model_args = ModelArgs(**get_model_config())
    model = RAFT(model_args)
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict)

    return RaftWrapper(model, model_args)
    
