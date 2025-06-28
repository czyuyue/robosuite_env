import copy
import pdb
from typing import List
import torch
import torch.nn as nn
from robosuite.modules.build import MODELS, build_model_from_cfg
from robosuite.modules.diffusion.pcd_transformer import TrajectoryInpaintingTransformer

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler



class KPDiff(nn.Module):
    def __init__(self,
                 model: TrajectoryInpaintingTransformer,
                 loss_args=None,
                 encoder_args=None,
                 decoder_args=None,
                 noise_scheduler=None,
                 **kwargs):
        super().__init__()

        self.loss_args = loss_args
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.query_embedding = torch.nn.Parameter(torch.randn(1, 1, 6), requires_grad=True)
        self.encoder = build_model_from_cfg(encoder_args)
        self.decoder = build_model_from_cfg(decoder_args)

    def get_context(self, pos, feat, query, n_b, n_q, text_emb=None, return_all=False):
        query_pos = torch.concat([pos, query], axis=-2)                                       # (B, N+Q, 3)
        query_feat = torch.concat([feat, self.query_embedding.repeat(n_b, n_q, 1)], axis=-2)  # (B, N+Q, 6)
        # if self.clip_fusion in ['early', 'both']:
        #     query_feat = torch.concat([query_feat, text_emb.unsqueeze(-2).repeat(1, query_feat.shape[1], 1)], axis=-1)          # (B, N+Q, 6+F_aligned_dim)
        p, f = self.encoder({'pos': query_pos, 'x': query_feat.transpose(1, 2)})
        if self.decoder is not None:
            f = self.decoder(p, f).squeeze(-1)   # (B, F, N+Q)

        if return_all is True:
            return f.transpose(1, 2)                 # (B, N+Q, F)
        else: 
            return f[:, :, -n_q:].transpose(1, 2)    # (B, Q, F)

    def forward(self, pos, feat, dtraj, query_mask, pack=None):
        n_b = dtraj.shape[0] # batch size
        n_q = dtraj.shape[1] # number of keypoints  
        keypoints = dtraj[:, :, 0, :] # (B, Q, 3)
        context = self.get_context(pos, feat, keypoints, n_b, n_q) # (B, Q, F)
        
        non_query_mask = 1 - query_mask # (B, Q)
        masked_dtraj = dtraj * non_query_mask # (B, Q, T, 3)
        
        # 拼接输入: [noisy_image, mask, masked_known_image]
        x = torch.cat([context, query_mask.unsqueeze(-1), masked_dtraj], dim=-1) # (B, Q, T, F+1+3)
        h = self.model(x)


        return h

    def inference(self, pos, feat, query, num_sample=10):

        pass

if __name__ == "__main__":
    pass
