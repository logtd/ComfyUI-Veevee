import os
import torch
from einops import rearrange

from folder_paths import models_dir

from ..utils.flow_viz import flow_to_color
from ..utils.flow_utils import get_full_frames_trajectories, flow_warp, forward_backward_consistency_check
from ..sea_raft.custom import get_model


RAFT_PATH = os.path.join(models_dir, 'raft')
os.makedirs(RAFT_PATH, exist_ok=True)


def get_raft_files():
    # print('wat')
    return os.listdir(RAFT_PATH)


def get_backward_occlusions(images, forward_flows, backward_flows):
    _, backward_occlusions = forward_backward_consistency_check(forward_flows, backward_flows)
    reshuffle_list = list(range(1,len(images)))+[0]
    warped_image1 = flow_warp(images, backward_flows)
    backward_occlusions = torch.clamp(backward_occlusions + (abs(images[reshuffle_list]-warped_image1).mean(dim=1)>255*0.25).float(), 0 ,1)
    return backward_occlusions


@torch.no_grad()
def pred_flows(flow_model, images):
    images = rearrange(images, 'b h w c -> b c h w').cuda() * 255.0
    
    reshuffle_list = list(range(1,len(images)))+[0]
    images_r = images[reshuffle_list]
    forward_flows, backward_flows = [], []

    for i in range(len(images)):
        forward_flow, _ = flow_model(images[i:i+1], images_r[i:i+1])
        forward_flows.append(forward_flow.to('cpu'))

    for i in range(len(images)):
        backward_flow, _= flow_model(images_r[i:i+1], images[i:i+1])
        backward_flows.append(backward_flow.to('cpu'))

    images = images.to('cpu')
    forward_flows = torch.cat(forward_flows, dim=0)
    backward_flows = torch.cat(backward_flows, dim=0)
    forward_occlusions = get_backward_occlusions(images, backward_flows, forward_flows)
    backward_occlusions = get_backward_occlusions(images, forward_flows, backward_flows)
    return {
        'forward': forward_flows,
        'backward': backward_flows,
        'forward_occlusions': forward_occlusions,
        'backward_occlusions': backward_occlusions
    }


def get_trajectories(images, backward_flows, backward_occlusions, scale=1):
    images = rearrange(images, 'b h w c -> b c h w')
    
    forward_trajectory, backward_trajectory, attn_masks = get_full_frames_trajectories(backward_flows, backward_occlusions, images, scale=8.0 * scale)
    
    trajectories = {}
    trajectories['forward_trajectory'] = forward_trajectory
    trajectories['backward_trajectory'] = backward_trajectory
    trajectories['attn_masks'] = attn_masks    
    
    return trajectories


class GetRaftFlowNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "checkpoint": (get_raft_files(),),
                "flow_type": (['SD15', 'SD15_Full', 'SDXL'],),
                "direction": (['forward', 'backward', 'both'],)
            }
        }

    RETURN_TYPES = ("FLOW","IMAGE", "IMAGE")

    FUNCTION = "process"
    CATEGORY = "vv/flow"

    # 
    def process(self, images, checkpoint, flow_type, direction):
        checkpoint_path = os.path.join(RAFT_PATH, checkpoint)
        flow_model = get_model(checkpoint_path)
        flow_model.eval()
        flow_model = flow_model.to('cuda')
        images = images.to('cuda')
        
        pred = pred_flows(flow_model, images)
        images = images.to('cpu')
        del flow_model
        
        forward_flow_imgs = flow_to_color(pred['forward'])
        forward_flow_imgs = rearrange(forward_flow_imgs, 'b c h w -> b h w c')  / 255.

        backward_flow_images = flow_to_color(pred['backward'])
        backward_flow_images = rearrange(backward_flow_images, 'b c h w -> b h w c')  / 255.
        model_type = 'SD1.5'
        get_forward_flow = direction == 'forward' or direction == 'both'
        get_backward_flow = direction == 'backward' or direction == 'both'
        forward_flows = []
        backward_flows = []
        if flow_type == 'SD15':
            if get_forward_flow:
                trajectories = get_trajectories(images, pred['backward'], pred['backward_occlusions'], 1)
                forward_flows.append(trajectories)
            if get_backward_flow:
                trajectories = get_trajectories(images, pred['forward'], pred['forward_occlusions'], 1)
                backward_flows.append(trajectories)
        elif flow_type == 'SD15_Full':
            for i in [1,2,4]:
                if get_forward_flow:
                    trajectories = get_trajectories(images, pred['backward'], pred['backward_occlusions'], i)
                    forward_flows.append(trajectories)
                if get_backward_flow:
                    trajectories = get_trajectories(images, pred['forward'], pred['forward_occlusions'], i)
                    backward_flows.append(trajectories)
        elif flow_type == 'SDXL':
            model_type = 'SDXL'
            for i in [2,4]:
                if get_forward_flow:
                    trajectories = get_trajectories(images, pred['backward'], pred['backward_occlusions'], i)
                    forward_flows.append(trajectories)
                if get_backward_flow:
                    trajectories = get_trajectories(images, pred['forward'], pred['forward_occlusions'], i)
                    backward_flows.append(trajectories)
        elif flow_type == 'SDXLT':
            model_type = 'SDXL'
            for i in [1, 2, 4]:
                if get_forward_flow:
                    trajectories = get_trajectories(images, pred['backward'], pred['backward_occlusions'], i)
                    forward_flows.append(trajectories)
                if get_backward_flow:
                    trajectories = get_trajectories(images, pred['forward'], pred['forward_occlusions'], i)
                    backward_flows.append(trajectories)
            
        trajectory = { 'forward_flows': forward_flows, 'backward_flows': backward_flows, 'model_type': model_type, 'direction': direction }
        torch.cuda.empty_cache()
        return (trajectory, forward_flow_imgs, backward_flow_images)
        