from typing import Optional, Union

import torch
from math import pi as PI


def get_color_wheel(device: torch.device) -> torch.Tensor:
    """
    Generates the color wheel.
    :param device: (torch.device) Device to be used
    :return: (torch.Tensor) Color wheel tensor of the shape [55, 3]
    """
    # Set constants
    RY: int = 15
    YG: int = 6
    GC: int = 4
    CB: int = 11
    BM: int = 13
    MR: int = 6
    # Init color wheel
    color_wheel: torch.Tensor = torch.zeros((RY + YG + GC + CB + BM + MR, 3), dtype=torch.float32)
    # Init counter
    counter: int = 0
    # RY
    color_wheel[0:RY, 0] = 255
    color_wheel[0:RY, 1] = torch.floor(255 * torch.arange(0, RY) / RY)
    counter: int = counter + RY
    # YG
    color_wheel[counter:counter + YG, 0] = 255 - torch.floor(255 * torch.arange(0, YG) / YG)
    color_wheel[counter:counter + YG, 1] = 255
    counter: int = counter + YG
    # GC
    color_wheel[counter:counter + GC, 1] = 255
    color_wheel[counter:counter + GC, 2] = torch.floor(255 * torch.arange(0, GC) / GC)
    counter: int = counter + GC
    # CB
    color_wheel[counter:counter + CB, 1] = 255 - torch.floor(255 * torch.arange(CB) / CB)
    color_wheel[counter:counter + CB, 2] = 255
    counter: int = counter + CB
    # BM
    color_wheel[counter:counter + BM, 2] = 255
    color_wheel[counter:counter + BM, 0] = torch.floor(255 * torch.arange(0, BM) / BM)
    counter: int = counter + BM
    # MR
    color_wheel[counter:counter + MR, 2] = 255 - torch.floor(255 * torch.arange(MR) / MR)
    color_wheel[counter:counter + MR, 0] = 255
    # To device
    color_wheel: torch.Tensor = color_wheel.to(device)
    return color_wheel


def _flow_hw_to_color(flow_vertical: torch.Tensor, flow_horizontal: torch.Tensor,
                      color_wheel: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Private function applies the flow color wheel to flow components (vertical and horizontal).
    :param flow_vertical: (torch.Tensor) Vertical flow of the shape [height, width]
    :param flow_horizontal: (torch.Tensor) Horizontal flow of the shape [height, width]
    :param color_wheel: (torch.Tensor) Color wheel tensor of the shape [55, 3]
    :param: device: (torch.device) Device to be used
    :return: (torch.Tensor) Visualized flow of the shape [3, height, width]
    """
    # Get shapes
    _, height, width = flow_vertical.shape
    # Init flow image
    flow_image: torch.Tensor = torch.zeros(3, height, width, dtype=torch.float32, device=device)
    # Get number of colors
    number_of_colors: int = color_wheel.shape[0]
    # Compute norm, angle and factors
    flow_norm: torch.Tensor = (flow_vertical ** 2 + flow_horizontal ** 2).sqrt()
    angle: torch.Tensor = torch.atan2(- flow_vertical, - flow_horizontal) / PI
    fk: torch.Tensor = (angle + 1.) / 2. * (number_of_colors - 1.)
    k0: torch.Tensor = torch.floor(fk).long()
    k1: torch.Tensor = k0 + 1
    k1[k1 == number_of_colors] = 0
    f: torch.Tensor = fk - k0
    # Iterate over color components
    for index in range(color_wheel.shape[1]):
        # Get component of all colors
        tmp: torch.Tensor = color_wheel[:, index]
        # Get colors
        color_0: torch.Tensor = tmp[k0] / 255.
        color_1: torch.Tensor = tmp[k1] / 255.
        # Compute color
        color: torch.Tensor = (1. - f) * color_0 + f * color_1
        # Get color index
        color_index: torch.Tensor = flow_norm <= 1
        # Set color saturation
        color[color_index] = 1 - flow_norm[color_index] * (1. - color[color_index])
        color[~color_index] = color[~color_index] * 0.75
        # Set color in image
        flow_image[index] = torch.floor(255 * color)
    return flow_image


def flow_to_color(flow: torch.Tensor, clip_flow: Optional[Union[float, torch.Tensor]] = None, 
                  normalize_over_video: bool = False) -> torch.Tensor:
    """
    Function converts a given optical flow map into the classical color schema.
    :param flow: (torch.Tensor) Optical flow tensor of the shape [batch size (optional), 2, height, width].
    :param clip_flow: (Optional[Union[float, torch.Tensor]]) Max value of flow values for clipping (default None).
    :param normalize_over_video: (bool) If true scale is normalized over the whole video (batch).
    :return: (torch.Tensor) Flow visualization (float tensor) with the shape [batch size (if used), 3, height, width].
    """
    # Check parameter types
    assert torch.is_tensor(flow), "Given flow map must be a torch.Tensor, {} given".format(type(flow))
    assert torch.is_tensor(clip_flow) or isinstance(clip_flow, float) or clip_flow is None, \
        "Given clip_flow parameter must be a float, a torch.Tensor, or None, {} given".format(type(clip_flow))
    # Check shapes
    assert flow.ndimension() in [3, 4], \
        "Given flow must be a 3D or 4D tensor, given tensor shape {}.".format(flow.shape)
    if torch.is_tensor(clip_flow):
        assert clip_flow.ndimension() == 0, \
            "Given clip_flow tensor must be a scalar, given tensor shape {}.".format(clip_flow.shape)
    # Manage batch dimension
    batch_dimension: bool = True
    if flow.ndimension() == 3:
        flow = flow[None]
        batch_dimension: bool = False
    # Save shape
    batch_size, _, height, width = flow.shape
    # Check flow dimension
    assert flow.shape[1] == 2, "Flow dimension must have the shape 2 but tensor with {} given".format(flow.shape[1])
    # Save device
    device: torch.device = flow.device
    # Clip flow if utilized
    if clip_flow is not None:
        flow = flow.clip(max=clip_flow)
    # Get horizontal and vertical flow
    flow_vertical: torch.Tensor = flow[:, 0:1]
    flow_horizontal: torch.Tensor = flow[:, 1:2]
    # Get max norm of flow
    flow_max_norm: torch.Tensor = (flow_vertical ** 2 + flow_horizontal ** 2).sqrt().view(batch_size, -1).max(dim=-1)[0]
    flow_max_norm: torch.Tensor = flow_max_norm.view(batch_size, 1, 1, 1)
    if normalize_over_video:
        flow_max_norm: Tensor = flow_max_norm.max(dim=0, keepdim=True)[0]
    # Normalize flow
    flow_vertical: torch.Tensor = flow_vertical / (flow_max_norm + 1e-05)
    flow_horizontal: torch.Tensor = flow_horizontal / (flow_max_norm + 1e-05)
    # Get color wheel
    color_wheel: torch.Tensor = get_color_wheel(device=device)
    # Init flow image
    flow_image = torch.zeros(batch_size, 3, height, width, device=device)
    # Iterate over batch dimension
    for index in range(batch_size):
        flow_image[index] = _flow_hw_to_color(flow_vertical=flow_vertical[index],
                                              flow_horizontal=flow_horizontal[index], color_wheel=color_wheel,
                                              device=device)
    return flow_image if batch_dimension else flow_image[0]