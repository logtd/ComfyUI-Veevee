from einops import rearrange
import torch
import torch.nn.functional as F

from ..gmflow.gmflow.geometry import flow_warp


def calc_mean_std(feat, eps=1e-5, chunk=1):
    size = feat.size()
    assert (len(size) == 4)
    if chunk == 2:
        feat = torch.cat(feat.chunk(2), dim=3)
    N, C = size[:2]
    feat_var = feat.view(N//chunk, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N//chunk, C, -1).mean(dim=2).view(N//chunk, C, 1, 1)
    return feat_mean.repeat(chunk,1,1,1), feat_std.repeat(chunk,1,1,1)


def adaptive_instance_normalization(content_feat, style_feat, chunk=1):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat, chunk)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def optimize_feature(idx, sub_idxs, sample, sampling_config, correlation_matrix=[], 
                     unet_chunk_size=2):
    
    """
    FRESO-guided latent feature optimization
    * optimize spatial correspondence (match correlation_matrix)
    * optimize temporal correspondence (match warped_image)
    """
    with torch.inference_mode(False):
        with torch.enable_grad():
            intra_weight = sampling_config.intra_weight
            spatial_iters = sampling_config.optimize_spatial_iters
            temporal_iters = sampling_config.optimize_temporal_iters
            flows, occs, _, _ = sampling_config.flow
            flows = [flows[0][sub_idxs], flows[1][sub_idxs]]
            occs = [occs[0][sub_idxs], occs[1][sub_idxs]]

            if (flows is None or occs is None) and (intra_weight == 0 or len(correlation_matrix) == 0):
                return sample
            # flows=[fwd_flows, bwd_flows]: (N-1)*2*H1*W1
            # occs=[fwd_occs, bwd_occs]: (N-1)*H1*W1
            # sample: 2N*C*H*W
            torch.cuda.empty_cache()
            video_length = sample.shape[0] // unet_chunk_size
            latent = rearrange(sample.to(torch.float32), "(b f) c h w -> b f c h w", f=video_length)
            
            cs = torch.nn.Parameter((latent.detach().clone()), requires_grad=True)
            optimizer = torch.optim.Adam([cs], lr=0.2)

            # unify resolution
            if flows is not None and occs is not None:
                scale = sample.shape[2] * 1.0 / flows[0].shape[2]
                kernel = int(1 / scale)
                bwd_flow_ = F.interpolate(flows[1] * scale, scale_factor=scale, mode='bilinear').repeat(unet_chunk_size,1,1,1)
                bwd_occ_ = F.max_pool2d(occs[1].unsqueeze(1), kernel_size=kernel).repeat(unet_chunk_size,1,1,1) # 2(N-1)*1*H1*W1
                fwd_flow_ = F.interpolate(flows[0] * scale, scale_factor=scale, mode='bilinear').repeat(unet_chunk_size,1,1,1)
                fwd_occ_ = F.max_pool2d(occs[0].unsqueeze(1), kernel_size=kernel).repeat(unet_chunk_size,1,1,1) # 2(N-1)*1*H1*W1
                # match frame 0,1,2,3 and frame 1,2,3,0
                reshuffle_list = list(range(1,video_length))+[0]
                
            # attention_probs is the GRAM matrix of the normalized feature 
            attention_probs = None
            if idx in correlation_matrix:
                attention_probs = correlation_matrix[idx]
            
            n_iter=[0]
            while n_iter[0] < max(spatial_iters, temporal_iters):
                def closure():
                    optimizer.zero_grad()
                    cs1 = cs.requires_grad_(True)
                    cs2 = cs[:,reshuffle_list].requires_grad_(True)
                    
                    loss = 0

                    # temporal consistency loss 
                    if flows is not None and occs is not None and n_iter[0] < temporal_iters:
                        c1 = rearrange(cs1, "b f c h w -> (b f) c h w")
                        c2 = rearrange(cs2, "b f c h w -> (b f) c h w")
                        warped_image1 = flow_warp(c1, bwd_flow_)
                        warped_image2 = flow_warp(c2, fwd_flow_)
                        loss = ((abs((c2-warped_image1)*(1-bwd_occ_)) + abs((c1-warped_image2)*(1-fwd_occ_))).mean() * 2)
                        
                    # spatial consistency loss
                    if attention_probs is not None and intra_weight > 0 and n_iter[0] < spatial_iters:
                        cs_vector = rearrange(cs, "b f c h w -> (b f) (h w) c")
                        #attention_scores = torch.bmm(cs_vector, cs_vector.transpose(-1, -2))
                        #cs_attention_probs = attention_scores.softmax(dim=-1)
                        cs_vector = cs_vector / ((cs_vector ** 2).sum(dim=2, keepdims=True) ** 0.5)
                        cs_attention_probs = torch.bmm(cs_vector, cs_vector.transpose(-1, -2))
                        tmp = (F.l1_loss(cs_attention_probs, attention_probs) * intra_weight)
                        loss = (tmp + loss)
                        
                    loss.backward()
                    n_iter[0]+=1
                
                    # if True: # for debug
                    #     print('Iteration: %d, loss: %f'%(n_iter[0]+1, loss.data.mean()))
                    return loss
                optimizer.step(closure)

            torch.cuda.empty_cache()
            return adaptive_instance_normalization(rearrange(cs.data.to(sample.dtype), "b f c h w -> (b f) c h w"), sample)