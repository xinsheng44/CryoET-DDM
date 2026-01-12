import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class HybridTransformerBlock(nn.Module):
    """Transformer-CNN hybrid module"""
    def __init__(self, in_channels, num_heads=4, expansion=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_channels)
        self.attn = nn.MultiheadAttention(in_channels, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(in_channels)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels * expansion),
            nn.GELU(),
            nn.Linear(in_channels * expansion, in_channels)
        )
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(4, in_channels),
            nn.GELU()
        )

    def forward(self, x):
        # Input x: [B, C, H, W]
        B, C, H, W = x.shape
        
        # Transformer branch
        x_trans = rearrange(x, 'b c h w -> b (h w) c')
        x_trans = self.norm1(x_trans)
        attn_out, _ = self.attn(x_trans, x_trans, x_trans)
        x_trans = x_trans + attn_out
        x_trans = x_trans + self.mlp(self.norm2(x_trans))
        x_trans = rearrange(x_trans, 'b (h w) c -> b c h w', h=H, w=W)
        
        # CNN branch
        x_conv = self.conv_block(x)
        
        # Fused output
        return x_trans + x_conv


class MultiScaleFeatureFusion(nn.Module):
    """Multi-scale feature fusion module"""
    def __init__(self, channels):
        super().__init__()
        self.conv1x1 = nn.ModuleList([
            nn.Conv2d(channels, channels//4, 1) for _ in range(4)
        ])
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(4, channels),
            nn.GELU()
        )
        
    def forward(self, features):
        # features: list of 4 feature maps at different scales
        resized = []
        target_size = features[0].shape[2:]
        
        for i, feat in enumerate(features):
            if i == 0:
                resized.append(self.conv1x1[i](feat))
            else:
                resized.append(F.interpolate(
                    self.conv1x1[i](feat), 
                    size=target_size, 
                    mode='bilinear',
                    align_corners=True
                ))
        
        fused = torch.cat(resized, dim=1)
        return self.fusion(fused)


class HybridUTransformer(nn.Module):
    """Hybrid CNN-Transformer denoising network"""
    def __init__(self, nf=48, base_width=11, top_width=3):
        super().__init__()
        
        
        self.ms_fusion = MultiScaleFeatureFusion(nf)
        
        # Encoder (downsampling path)
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, nf, base_width, padding=base_width//2),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2)
        )
        self.enc2 = self._make_enc_layer(nf, nf)
        self.enc3 = self._make_enc_layer(nf, nf)
        self.enc4 = self._make_enc_layer(nf, nf)
        self.enc5 = self._make_enc_layer(nf, nf)
        
        # Bottleneck layer - using Transformer modules
        self.bottleneck = nn.Sequential(
            nn.Conv2d(nf, nf*2, 3, padding=1),
            HybridTransformerBlock(nf*2),
            HybridTransformerBlock(nf*2),
            nn.Conv2d(nf*2, nf, 3, padding=1)
        )
        
        # Decoder (upsampling path)
        self.dec5 = self._make_dec_layer(nf*2, nf*2)
        self.dec4 = self._make_dec_layer(nf*3, nf*2)
        self.dec3 = self._make_dec_layer(nf*3, nf*2)
        self.dec2 = self._make_dec_layer(nf*3, nf*2)
        
        # Output layer
        self.out = nn.Sequential(
            nn.Conv2d(nf*2+1, 64, top_width, padding=top_width//2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 32, top_width, padding=top_width//2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 1, top_width, padding=top_width//2)
        )
        
    def _make_enc_layer(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2)
        )
    
    def _make_dec_layer(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.LeakyReLU(0.1)
        )
    
    def forward(self, x):
        # Encoding path
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        
        # Bottleneck layer
        fused = self.ms_fusion([e2, e3, e4, e5])
        b = self.bottleneck(fused)
        
        # Decoding path
        d5 = F.interpolate(b, size=e4.shape[2:], mode='bilinear', align_corners=True)
        d5 = torch.cat([d5, e4], 1)
        d5 = self.dec5(d5)
        
        d4 = F.interpolate(d5, size=e3.shape[2:], mode='bilinear', align_corners=True)
        d4 = torch.cat([d4, e3], 1)
        d4 = self.dec4(d4)
        
        d3 = F.interpolate(d4, size=e2.shape[2:], mode='bilinear', align_corners=True)
        d3 = torch.cat([d3, e2], 1)
        d3 = self.dec3(d3)
        
        d2 = F.interpolate(d3, size=e1.shape[2:], mode='bilinear', align_corners=True)
        d2 = torch.cat([d2, e1], 1)
        d2 = self.dec2(d2)
        
        # Output layer
        d1 = F.interpolate(d2, size=x.shape[2:], mode='bilinear', align_corners=True)
        d1 = torch.cat([d1, x], 1)
        return self.out(d1)