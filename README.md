# ComfyUI-Veevee (Work in progress)

A Video2Video framework for text2image models in ComfyUI. Supports SD1.5 and SDXL.

## Examples
(TODO more examples)
See `example_workflows` directory for SD15 and SDXL examples with notes.

## Installation 
Install this repo from the ComfyUI manager or git clone the repo into `custom_nodes` then `pip install -r requirements.txt` within the cloned repo.

### Required Models
It is recommended to use Flow Attention through Unimatch (and others soon). 
To get Unimatch optical flow go to `https://github.com/autonomousvision/unimatch/blob/master/MODEL_ZOO.md#optical-flow` and select one of the models. **It must be one of the regrefine larger models**.
Sintel and things versions tend to give the best results.
  
## Nodes

### VV] Apply Model
This node takes your diffusion model (SD15/SDXL) and alters the attentions, blocks, and resnets.
The change is permanent to the model you input and will affect the model even when the output of the node is not used.
However, Veevee will not activate unless using the VV Unsampler or VV Sampler.

### VV] Unsampler
This node coordinates unsampling the source video into unsampled noise.

#### Inputs
* overlap: If you use this with AnimateDiff-Evolved's batching, this should match the context overlap otherwise you will get misconfigured injections.
* (optional) flow_config: This is the configuration to use flow attention, it is recommended.
* (optional) inj_config: This is the injection config to store information about the unsampling process that can be utilized during sampling. It is recommended.
* (optional) sampler: Use a sampler that's not in default ComfyUI.

### VV] Sampler
This node coordinates sampling the unsampled noise into your target generation.

#### Inputs
* overlap: If you use this with AnimateDiff-Evolved's batching, this should match the context overlap otherwise you will get misconfigured injections.
* sampler_name: The sampler to use if optional sampler is not specific.
* attn_injection_steps: The steps of attention to inject from the unsampling process. Can not be greater than the inj config's saved attention steps.
* res_injection_steps: The steps of res to inject from the unsampling process. Can not be greater than the inj config's saved res steps.
* (optional) flow_config: This is the configuration to use flow attention, it is recommended.
* (optional) inj_config: This is the injection config to utilize the stored information from unsampling. It must be the same config as the one used during unsampling. It is recommended.
* (optional) sca_config: This is to use Sparse Casual Attention. It can help is certain use cases.
* (optional) pivot_config: This is to use Pivot Attention.
* (optional) rave_config: This is to use Rave Attention.
* (optional) temporal_config: You can control the steps that AnimateDiff (AD) runs using this config. If you do not specify a config AD will run as normal.
* (optional) sampler: Use a sampler that's not in default ComfyUI.

### VV] Injection Config
This node sets up injection parameters and shares information between the unsampler and sampler. Must use the same node between the two.

#### Inputs
* unsampler_sigmas: The sigmas used from scheduling the unsampler.
* sampler_sigmas: The sigmas used from scheduling the sampler.
* save_attn_steps: The amount of steps to save from the attention.
* save_res_steps: The amount of steps to save from the resnets.

### VV] Get Flow
This node calculates trajectories to guide Flow Attention.

#### Inputs
* images: Your input video frames.
* checkpoint: The unimatch checkpoint that must be in `ComfyUI/models/unimatch`
* flow_type: There are three flow types available. These must match your model.
  * SD15 is the standard flow for SD15
  * SD15_Full utilizes a stronger flow for SD15
  * SDXL is the standard flow for SDXL
* direction: The direction to use in flow attention.

### VV] Flow Attn Config
Utilize flow attention by configuring this for your unsampler/sampler.

#### Inputs
* flow: The output from the Get Flow node.
* targets: Which parts of the UNet should utilize this attention.
* start_percent and end_percent are the step range

### VV] Sparse Casual Attn Config
SCA Attention allows attentions to from specific frames to look at other frames.

#### Inputs
* direction: The direction which the attentions can look in terms of video frames.
* targets: Which parts of the UNet should utilize this attention.
* start_percent and end_percent are the step range

### VV] Pivot Attn Config
Pivot attention is variant of the pivot mechanism in TokenFlow.
It selects batches from the given frames and selects a "pivot" frame for each batch randomly.
This frame is used to select styles for all frames within the batch.

#### Inputs
* batch_size: The batch size to cut the frames into.
* seed: A random seed for selecting batch pivots.
* targets: Which parts of the UNet should utilize this attention.
* start_percent and end_percent are the step range

### VV] Rave Attn Config
RAVE gridifies frames before applying the attention mechanism to allow styles to diffuse.

#### Inputs
* grid_size: The length of the grid. For example 2 gives a 2x2 grid.
* seed: A random seed for selecting batch pivots.
* targets: Which parts of the UNet should utilize this attention.
* start_percent and end_percent are the step range

### VV] Temporal Attn Config
This node allows some more control over running AnimateDiff alongside Veevee.
**Note: AnimateDiff must be ran at a low effect multival if used.**

#### Inputs
* start_percent and end_percent are the step range

## Acknowledgements
(TODO)
FLATTEN
FRESCO
Rave
Video2Video-zero
Unimatch
FlowDiffuser
CoTracker
