import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# # use the following lines to import when running this .py file independently
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from layers import *
# from warplayer import *

from .layers import *  
from .warplayer import warp

from utils.general import multi_scale

from pdb import set_trace as stx
import numbers
from einops import rearrange
import math


class RMSNormPerHead(nn.Module):
    """RMS normalization applied per attention head on channel axis (without bias)."""
    def __init__(self, num_heads, head_channels, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.num_heads = num_heads
        self.head_channels = head_channels  # number of channels per head (2*c if V is 2c)
        self.weight = nn.Parameter(torch.ones(num_heads, head_channels))

    def forward(self, x):  
        # x: [B, H, 2c, N] or [B, H, 2c, H, W]
        B, H, C2, *rest = x.shape
        x_flat = x.view(B, H, C2, -1)  # flatten spatial dimensions
        x_norm = x_flat * torch.rsqrt(x_flat.pow(2).mean(dim=2, keepdim=True) + self.eps)
        x_norm = x_norm * self.weight.view(1, H, C2, 1)  # apply learnable per-head scale
        return x_norm.view(B, H, C2, *rest)

# def _lambda_schedule(l, base=0.8):
#     # Default λ initialization from paper: λ_init = 0.8 - 0.6 * exp(-0.3*(l-1))
#     return float(0.8 - 0.6 * math.exp(-0.3 * (l - 1)))

class DTAM(nn.Module):
    """
    Differential Transposed Attention Module.
    
    Key points:
      - Split Q/K into two parts (Q1,K1) and (Q2,K2), compute transposed attention separately, then subtract with λ scaling.
      - V has 2C channels
      - λ is reparameterized as λ = exp(λq1·λk1) - exp(λq2·λk2) + λ_init (shared across heads in the same layer).
      - Each head output is normalized with head-wise RMSNorm and multiplied by (1 - λ_init) to stabilize gradient.
    """
    def __init__(self, channels, num_heads=2, layer_index=1, dw_kernel=3):
        super().__init__()
        assert channels % num_heads == 0, "Channels must be divisible by num_heads"
        self.C = channels
        self.num_heads = num_heads
        self.c = channels // num_heads  # channels per head
        pad = dw_kernel // 2
        self.temperature1 = nn.Parameter(torch.ones(num_heads, 1, 1)/math.sqrt(self.c)) 
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1)/math.sqrt(self.c)) 

        self.norm = LayerNorm(self.C, False)

        # Pointwise 1x1 convolution for Q, K, V
        self.q_pw = nn.Conv2d(self.C, 2*self.C, 1, bias=False)
        self.k_pw = nn.Conv2d(self.C, 2*self.C, 1, bias=False)
        self.v_pw = nn.Conv2d(self.C, 2*self.C, 1, bias=False)

        # Depthwise 3x3 convolution for local context (applied after 1x1)
        self.q_dw = nn.Conv2d(2*self.C, 2*self.C, dw_kernel, padding=pad, groups=2*self.C, bias=False)
        self.k_dw = nn.Conv2d(2*self.C, 2*self.C, dw_kernel, padding=pad, groups=2*self.C, bias=False)
        self.v_dw = nn.Conv2d(2*self.C, 2*self.C, dw_kernel, padding=pad, groups=2*self.C, bias=False)

        # RMSNorm applied per head on 2c channels
        self.head_norm = RMSNormPerHead(num_heads=self.num_heads, head_channels=2*self.c)

        # Output projection W_o: 2C -> C (maps concatenated heads back to original channels)
        self.proj_out = nn.Conv2d(2*self.C, self.C, 1, bias=False)

        # λ reparameterization (shared across heads; vector length = c)
        # self.lq1 = nn.Parameter(torch.zeros(self.c))
        # self.lk1 = nn.Parameter(torch.zeros(self.c))
        # self.lq2 = nn.Parameter(torch.zeros(self.c))
        # self.lk2 = nn.Parameter(torch.zeros(self.c))
        self.lq1 = nn.Parameter(torch.zeros(self.c, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lk1 = nn.Parameter(torch.zeros(self.c, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lq2 = nn.Parameter(torch.zeros(self.c, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lk2 = nn.Parameter(torch.zeros(self.c, dtype=torch.float32).normal_(mean=0, std=0.1))
        

        # initial λ value
        lam_init = 0.8
        self.register_buffer("lambda_init", torch.tensor(lam_init, dtype=torch.float32))

        # fixed factor (1 - λ_init) applied after RMSNorm to stabilize gradient
        self.register_buffer("one_minus_lambda_init", torch.tensor(1.0 - lam_init, dtype=torch.float32))

        self.logit_scale = nn.Parameter(torch.zeros(1))

    def _lambda_scalar(self):
        # λ = exp(λq1·λk1) - exp(λq2·λk2) + λ_init
        a = torch.exp(torch.sum(self.lq1 * self.lk1))
        b = torch.exp(torch.sum(self.lq2 * self.lk2))
        return a - b + self.lambda_init

    def forward(self, x):  # x: [B,C,H,W]
        B, C, H, W = x.shape
        N = H * W
        
        x_ = x
        x = self.norm(x)

        # Compute Q, K, V with local context (pointwise -> depthwise)
        q = self.q_dw(self.q_pw(x))  # [B,2C,H,W]
        k = self.k_dw(self.k_pw(x))
        v = self.v_dw(self.v_pw(x))  # [B,2C,H,W]

        # Reshape to separate heads
        q_all = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads) # [B,head,2c,N] 
        k_all = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads) # [B,head,2c,N]
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads) # [B,head,2c,N]

        # Split Q/K into two halves per head for differential attention
        q1, q2 = torch.split(q_all, self.c, dim=2)  # [B,Hd,c,N] x2
        k1, k2 = torch.split(k_all, self.c, dim=2)  # [B,Hd,c,N] x2

        # Compute channel-wise attention
        attn1 = torch.softmax((q1 @ k1.transpose(-2, -1)) * self.temperature1, dim=-1)
        attn2 = torch.softmax((q2 @ k2.transpose(-2, -1)) * self.temperature2, dim=-1)

        lam = self._lambda_scalar()
        attn = attn1 - lam * attn2  # differential attention

        v1, v2 = torch.split(v, self.c, dim=2)
        y1 = attn @ v1 # apply attention to V: [B,head,c,N]
        y2 = attn @ v2

        y = torch.cat([y1, y2], dim=2)

        # Head-wise RMSNorm followed by (1 - λ_init) scaling to stabilize gradients
        y = self.head_norm(y) * self.one_minus_lambda_init 

        # reshape back to [B,C,H,W] and apply output projection
        y = rearrange(y, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=H, w=W)
        out = self.proj_out(y)

        return out + x_
    
class ResBs(nn.Module):
    def __init__(self, ch, num_res=8, norm=False):

        super(ResBs, self).__init__()
        layers = []
        
        for _ in range(num_res):
            layers.append(ResBlock(ch, ch, norm=norm))

        self.rbs = nn.Sequential(*layers)

    def forward(self, x):
        return self.rbs(x)
    
class mf_af(nn.Module):
    def __init__(self, ch, out_ch, c=int, num_heads=int):
        super(mf_af, self).__init__()

        self.naf = NAFBlock(ch)
        self.naf_af = NAFBlock(ch)
        self.conv_af = BasicConv(ch, out_ch, kernel_size=3, stride=1)
        self.conv_mf = BasicConv(ch, out_ch, kernel_size=3, stride=1)
        self.conv_deep = nn.Conv2d(c,c,3,1,1)
        self.upsample = nn.PixelShuffle(2)
        self.diff_attn = DTAM(ch, num_heads)


    def forward(self, x, deep_af):
        if deep_af is not None: 
            deep_af = self.upsample(deep_af)
            deep_af = self.conv_deep(deep_af)
            x = torch.cat((x, deep_af), dim=1)  

        x_ = self.naf(x)

        af = self.conv_af(self.naf_af(x_))
        mf = self.conv_mf(self.diff_attn(x_))

        return af, mf
    
class mf2flow(nn.Module):
    def __init__(self, in_ch, ch):
        super(mf2flow, self).__init__()

        self.conv = nn.Sequential(
            BasicConv(in_ch, ch, kernel_size=3, stride=1, bias=True),
            BasicConv(ch, ch, kernel_size=3, stride=1, bias=True),
            BasicConv(ch, 5, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, mf, deep_flow):
        if deep_flow is not None:  
            flow = deep_flow[:,:4]
            flow = F.interpolate(flow, scale_factor = 2, mode="bilinear", align_corners=False) * 2
            x = torch.cat((mf, flow), dim=1)  
        else:
            x = mf

        x = self.conv(x)

        return x



class OMDNet(nn.Module):
    def __init__(self, num_res=16, norm=False, head = [1,2,4,8]):
        super(OMDNet, self).__init__()

        base_ch = 32

        self.Enconv1 = BasicConv(3, base_ch, kernel_size=3, relu=True, stride=1)
        self.EnRes1 = ResBs(base_ch, num_res, norm=norm)

        self.Enconv2 = BasicConv(base_ch, base_ch * 2, kernel_size=3, relu=True, stride=2)
        self.EnRes2 = ResBs(base_ch * 2, num_res, norm=norm)

        self.Enconv3 = BasicConv(base_ch * 2, base_ch * 4, kernel_size=3, relu=True, stride=2)
        self.EnRes3 = ResBs(base_ch * 4, num_res, norm=norm)

        self.midconv = BasicConv(base_ch * 4, base_ch * 8, kernel_size=3, relu=True, stride=2)
        self.midRes = ResBs(base_ch * 8, num_res, norm=norm)
        self.upconvmid = BasicConv(base_ch * 8, base_ch * 4, kernel_size=4, relu=True, stride=2, transpose=True, norm=norm)

        self.feature_extraction3 = mf_af(base_ch * 4, base_ch * 2, base_ch *2 //4, num_heads=head[3])

        self.Deconv3 = BasicConv(base_ch * 6, base_ch * 4, kernel_size=1, relu=True, stride=1)
        self.DeRes3 = ResBs(base_ch * 4, num_res, norm=norm)
        self.upconv3 = BasicConv(base_ch * 4, base_ch * 2, kernel_size=4, relu=True, stride=2, transpose=True, norm=norm)

        self.feature_extraction2 = mf_af(base_ch * 2 + base_ch *2 //4, base_ch * 2, base_ch *2 //4, num_heads=head[2])

        self.Deconv2 = BasicConv(base_ch * 4, base_ch * 2, kernel_size=1, relu=True, stride=1)
        self.DeRes2 = ResBs(base_ch * 2, num_res, norm=norm)
        self.upconv2 = BasicConv(base_ch * 2, base_ch, kernel_size=4, relu=True, stride=2, transpose=True, norm=norm)

        self.feature_extraction1 = mf_af(base_ch + base_ch*2//4, base_ch, base_ch*2//4, num_heads=head[1])

        self.Deconv1 = BasicConv(base_ch * 2, base_ch, kernel_size=1, relu=True, stride=1)
        self.DeRes1 = ResBs(base_ch, num_res, norm=norm)

        self.convsout3 = nn.Sequential(
            BasicConv(base_ch * 4, base_ch * 2, kernel_size=3, stride=1, relu=True, norm=norm),
            BasicConv(base_ch * 2, base_ch * 1, kernel_size=3, stride=1, relu=True, norm=norm),
            BasicConv(base_ch * 1, 3, kernel_size=3, stride=1, relu=False, norm=norm)
        )
        self.convsout2 = nn.Sequential(
            BasicConv(base_ch * 2, base_ch * 1, kernel_size=3, stride=1, relu=True, norm=norm),
            BasicConv(base_ch * 1, 3, kernel_size=3, stride=1, relu=False, norm=norm)
        )
        self.convsout1 = BasicConv(base_ch * 1, 3, kernel_size=3, relu=False, stride=1)

        self.drop1 = nn.Dropout2d(0.1)
        self.drop2 = nn.Dropout2d(0.1)

        self.flow3 = mf2flow(base_ch * 2, 32)
        self.flow2 = mf2flow(base_ch * 2 + 4, 16)
        self.flow1 = mf2flow(base_ch * 1 + 4, 8)

        self.gate1 = nn.Sequential(
            BasicConv(6, 8, kernel_size=3, relu=True, stride=1),
            BasicConv(8, 1, kernel_size=3, relu=False, stride=1)
        )
        self.gate2 = nn.Sequential(
            BasicConv(6, 8, kernel_size=3, relu=True, stride=1),
            BasicConv(8, 1, kernel_size=3, relu=False, stride=1)
        )
        self.gate3 = nn.Sequential(
            BasicConv(6, 8, kernel_size=3, relu=True, stride=1),
            BasicConv(8, 1, kernel_size=3, relu=False, stride=1)
        )

    
    def forward(self, x, gt, mask, train):
        x1 = x
        x2 = F.interpolate(x, scale_factor=0.5)
        x3 = F.interpolate(x2, scale_factor=0.5)

        gates = []
        deblured = []

        e1 = self.Enconv1(x1)
        e1 = self.EnRes1(e1)

        e2 = self.Enconv2(e1)
        e2 = self.EnRes2(e2)

        e3 = self.Enconv3(e2)
        e3 = self.EnRes3(e3)

        # ed_mid = self.midconv(e3)
        # ed_mid = self.midRes(ed_mid)
        # ed_up = self.upconvmid(ed_mid)

        af3, mf3 = self.feature_extraction3(e3, None)
        # d3 = torch.cat([ed_up, af3], dim=1)
        d3 = torch.cat([e3, af3], dim=1)

        d3 = self.Deconv3(d3)
        d3 = self.DeRes3(d3)
        d3_out = d3
        d3 = self.upconv3(d3)

        af2, mf2 = self.feature_extraction2(e2, af3)
        d2 = torch.cat([d3,af2], dim=1)

        d2 = self.Deconv2(d2)
        d2 = self.DeRes2(d2)
        d2_out = d2
        d2 = self.upconv2(d2)

        af1, mf1 = self.feature_extraction1(e1, af2)
        d1 = torch.cat([d2,af1], dim=1)

        d1 = self.Deconv1(d1)
        d1 = self.DeRes1(d1)
        d1_out = d1

        out1 = self.convsout1(d1_out) 
        out2 = self.convsout2(d2_out) 
        out3 = self.convsout3(d3_out) 

        whole_deblured1 = out1 + x1
        whole_deblured2 = out2 + x2
        whole_deblured3 = out3 + x3
        whole_deblured = []
        whole_deblured.extend([whole_deblured1, whole_deblured2, whole_deblured3])

        flow3 = self.flow3(mf3, None)
        flow2 = self.flow2(mf2, flow3)
        flow1 = self.flow1(mf1, flow2)

        flow1_mag, mag_diff1 = self.flow_mag(flow1[:, :4])
        flow2_mag, mag_diff2 = self.flow_mag(flow2[:, :4])
        flow3_mag, mag_diff3 = self.flow_mag(flow3[:, :4])
        gate1 = F.sigmoid(self.gate1(flow1_mag))
        gate2 = F.sigmoid(self.gate2(flow2_mag))
        gate3 = F.sigmoid(self.gate3(flow3_mag))
        
        gates.extend([gate1, gate2, gate3]) 
        # mag_diffs = []
        # mag_diffs.extend([mag_diff1, mag_diff2, mag_diff3])

        if train:
            merged = []
            wrap_first = []
            wrap_last = []

            [gt1, gt2, gt3] = multi_scale(gt)
            [mask1, mask2, mask3] = multi_scale(mask)

            warped_first_frame1 = warp(gt1, flow1[:, :2]) 
            warped_last_frame1 = warp(gt1, flow1[:, 2:4]) 
            merged1 = (warped_first_frame1 * torch.sigmoid(flow1[:, 4:5]) + warped_last_frame1 * (1 - torch.sigmoid(flow1[:, 4:5])))*2/3 + (whole_deblured1)/3 

            warped_first_frame2 = warp(gt2, flow2[:, :2]) 
            warped_last_frame2 = warp(gt2, flow2[:, 2:4]) 
            merged2 = (warped_first_frame2 * torch.sigmoid(flow2[:, 4:5]) + warped_last_frame2 * (1 - torch.sigmoid(flow2[:, 4:5])))*2/3 + (whole_deblured2)/3 

            warped_first_frame3 = warp(gt3, flow3[:, :2]) 
            warped_last_frame3 = warp(gt3, flow3[:, 2:4]) 
            merged3 = (warped_first_frame3 * torch.sigmoid(flow3[:, 4:5]) + warped_last_frame3 * (1 - torch.sigmoid(flow3[:, 4:5])))*2/3 + (whole_deblured3)/3

            merged.extend([merged1, merged2, merged3])
            wrap_first.extend([warped_first_frame1, warped_first_frame2, warped_first_frame3])
            wrap_last.extend([warped_last_frame1, warped_last_frame2, warped_last_frame3])
           
            deblured1_ = whole_deblured1 * gate1 + (mask1*x1+(1-mask1)*gt1) * (1-gate1) 
            deblured2_ = whole_deblured2 * gate2 + (mask2*x2+(1-mask2)*gt2) * (1-gate2)
            deblured3_ = whole_deblured3 * gate3 + (mask3*x3+(1-mask3)*gt3) * (1-gate3)
            deblured1 = whole_deblured1 * gate1 + x1 * (1-gate1)
            deblured2 = whole_deblured2 * gate2 + x2 * (1-gate2)
            deblured3 = whole_deblured3 * gate3 + x3 * (1-gate3)
            deblured_ = []
            deblured.extend([deblured1, deblured2, deblured3])
            deblured_.extend([deblured1_, deblured2_, deblured3_])

            return whole_deblured, deblured, deblured_, gates, wrap_first, wrap_last, merged

        else:
            deblured1 = whole_deblured1 * gate1 + x1 * (1-gate1)
            deblured2 = whole_deblured2 * gate2 + x2 * (1-gate2)
            deblured3 = whole_deblured3 * gate3 + x3 * (1-gate3)

            deblured.extend([deblured1, deblured2, deblured3])

            return whole_deblured, deblured, deblured, gates, None, None, None
    
    def flow_mag(self,flow):
        # flow: [B, 4, H, W]
        u1, v1 = flow[:, 0], flow[:, 1]  
        u2, v2 = flow[:, 2], flow[:, 3]  
        # compute magnitude, keep dimension [B,1,H,W]
        mag1 = torch.sqrt(u1 ** 2 + v1 ** 2).unsqueeze(1)
        mag2 = torch.sqrt(u2 ** 2 + v2 ** 2).unsqueeze(1)
        # cat -> [B, 6, H, W]
        flow6 = torch.cat([flow, mag1, mag2], dim=1)
        #
        mag_diff = (mag1.mean(dim=[1,2,3]) - mag2.mean(dim=[1,2,3]))  # [B]
        return flow6, mag_diff
    
if __name__ == "__main__":

    model = OMDNet(num_res=20, num_attn=0)


    # # --------- FLOPs and Params ----------
    # import torch
    # from thop import profile
    # model.eval()
    # a = torch.randn(1, 3, 256, 256).to(next(model.parameters()).device)

    # flops, params = profile(model, inputs=(a, a, a, True), verbose=False)                

    # print(f"FLOPs: {flops/1e9:.4f} G")   
    # print(f"Params: {params/1e6:.2f} M")



    # # ---------- inference time ------------
    # import time
    # import torch
    # import numpy as np
    # def inference_time_ms(model, inputs, warmup=20, runs=100):
    #     """
    #         forward inderence time only
    #     """
    #     device = next(model.parameters()).device
    #     model.eval()

    #     @torch.inference_mode()
    #     def fwd():
    #         return model(*inputs) if isinstance(inputs, (tuple, list)) else model(inputs)

    #     # warmup
    #     if device.type == "cuda":
    #         torch.cuda.synchronize()
    #         for _ in range(warmup):
    #             _ = fwd()
    #         torch.cuda.synchronize()

    #         # CUDA events 
    #         start = torch.cuda.Event(enable_timing=True)
    #         end   = torch.cuda.Event(enable_timing=True)
    #         times = []
    #         for _ in range(runs):
    #             start.record()
    #             _ = fwd()
    #             end.record()
    #             torch.cuda.synchronize()
    #             times.append(start.elapsed_time(end))  # ms
    #     else:
    #         # CPU 
    #         for _ in range(warmup):
    #             _ = fwd()
    #         times = []
    #         for _ in range(runs):
    #             t0 = time.perf_counter()
    #             _ = fwd()
    #             t1 = time.perf_counter()
    #             times.append((t1 - t0) * 1000.0)

    #     return float(np.mean(times))  # ms
    
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = model.to(device).eval()                 
    # torch.backends.cudnn.benchmark = True          
    # x = torch.randn(1, 3, 1920, 360, device=device)
    # avg_ms_eval  = inference_time_ms(model, inputs=(x, x, x, False))

    # print(f"Inference time: {avg_ms_eval:.2f} ms")


# python models/deblur_model.py