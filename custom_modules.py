import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from lvdm.modules.networks.ae_modules import Decoder, nonlinearity

class CustomVideoDataset(Dataset):
    def __init__(self, data_dir, logo_dir, img_size=256, num_frames=8):
        self.data_dir = data_dir
        self.logo_dir = logo_dir
        self.img_size = img_size
        self.num_frames = num_frames
        
        self.video_folders = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                              if os.path.isdir(os.path.join(data_dir, f))]

        self.logo_images = []
        for root, dirs, files in os.walk(logo_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                    self.logo_images.append(os.path.join(root, file))

        self.transform_video = transforms.Compose([
            transforms.Resize((320, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]), 
        ])
        self.transform_image = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]), 
        ])

    def __len__(self):
        return len(self.video_folders)

    def __getitem__(self, idx):
        video_folder = self.video_folders[idx]

        frames = []
        for i in range(self.num_frames):
            frame_path = os.path.join(video_folder, f"frame_{i}.jpg")
            frame = Image.open(frame_path).convert('RGB')
            frame = self.transform_video(frame)  
            frames.append(frame)

        random_logo_path = np.random.choice(self.logo_images)
        logo_image = Image.open(random_logo_path).convert('RGB')
        logo_image = self.transform_image(logo_image)  
        frames_tensor = torch.stack(frames, dim=0)

        return frames_tensor, logo_image

class CustomVideoLatentDataset(Dataset):
    def __init__(self, data_dir, logo_dir, img_size=256, num_frames=8):
        self.data_dir = data_dir
        self.logo_dir = logo_dir
        self.img_size = img_size
        self.num_frames = num_frames
        
        self.video_folders = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
        
        self.logo_images = []
        for root, dirs, files in os.walk(logo_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                    self.logo_images.append(os.path.join(root, file))
        
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]), 
        ])

    def __len__(self):
        return len(self.video_folders)

    def __getitem__(self, idx):
        
        video_latent = torch.load(self.video_folders[idx], map_location='cpu')

        random_logo_path = np.random.choice(self.logo_images)
        logo_image = Image.open(random_logo_path).convert('RGB')
        logo_image = self.transform(logo_image)  

        return video_latent, logo_image

from SFMamba_2D import SFMambaUNet

class EmbeddingNet(Decoder):
    def __init__ (self, batch_size=1, decoder_weights=None,  **kwargs) :
        super().__init__(**kwargs)
        self.adapter = SFMambaUNet(batch_size=batch_size)
        self.load_state_dict(decoder_weights, strict=False)

        for param in self.parameters():
            param.requires_grad = False
        for param in self.adapter.parameters():
            param.requires_grad = True

    def forward(self, z, condition_image=None):
        vae_features = []
        vae_features.append(z)

        self.last_z_shape = z.shape
        temb = None

        h = self.conv_in(z)
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        
        for i_level in reversed(range(self.num_resolutions)):
            if i_level == 1 or i_level == 2:
                vae_features.append(h)

            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)

            if i_level != 0:
                h = self.up[i_level].upsample(h)
        
        if self.give_pre_end:
            return h
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        multi_scale_watermark = []
        multi_scale_watermark.append(F.interpolate(condition_image, size=(320, 512), mode='bilinear', align_corners=False))
        multi_scale_watermark.append(F.interpolate(condition_image, size=(160, 256), mode='bilinear', align_corners=False))
        multi_scale_watermark.append(F.interpolate(condition_image, size=(80, 128), mode='bilinear', align_corners=False))
        multi_scale_watermark.append(F.interpolate(condition_image, size=(40, 64), mode='bilinear', align_corners=False))

        input_of_encoder = torch.cat((multi_scale_watermark[0], h), dim=1)
        h = self.adapter(input_of_encoder, vae_features, multi_scale_watermark)
        
        return h if not self.tanh_out else torch.tanh(h)

from SFMamba_3D import SFMamba_3D
from einops.layers.torch import Rearrange
from DWT_3d_index import wavelet_3d_index

class RevealNet(nn.Module):
    def __init__(self, batch_size, nc=3, nhf=24, output_function=nn.Tanh):
        super().__init__()

        self.feat_extract = nn.Sequential(
            Rearrange('n d c h w -> n c d h w'),
            nn.Conv3d(nc, nhf, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            Rearrange('n c d h w -> n d c h w'))

        self.block1 = nn.Sequential(
            SFMamba_3D(dim=nhf, scan_index = self.wavelet_index(torch.ones(batch_size, 8, 3, 320, 512))),
            Rearrange('n d c h w -> n c d h w'),
            nn.Conv3d(nhf, nhf, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.ReLU(True),
            Rearrange('n c d h w -> n d c h w'))

        self.block2 = nn.Sequential(
            SFMamba_3D(dim=nhf, scan_index = self.wavelet_index(torch.ones(batch_size, 8, 3, 160, 256))),
            Rearrange('n d c h w -> n c d h w'),
            nn.Conv3d(nhf, nhf, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.ReLU(True),
            Rearrange('n c d h w -> n d c h w'))

        self.block3 = nn.Sequential(
            SFMamba_3D(dim=nhf, scan_index = self.wavelet_index(torch.ones(batch_size, 8, 3, 80, 128))),
            Rearrange('n d c h w -> n c d h w'),
            nn.Conv3d(nhf, nhf, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(True),
            Rearrange('n c d h w -> n d c h w'))

        self.block4 = nn.Sequential(
            SFMamba_3D(dim=nhf, scan_index = self.wavelet_index(torch.ones(batch_size, 8, 3, 80, 128))),
            Rearrange('n d c h w -> n c d h w'),
            nn.Conv3d(nhf, 4, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.Upsample(size=(8, 64, 128), mode='trilinear', align_corners=False), 
            Rearrange('n c d h w -> n d c h w'),
            output_function())  


    def wavelet_index(self, x):
        B, nf, C, H, W = x.shape
        idx_tensor = wavelet_3d_index(nf, H, W)
        return idx_tensor

    def forward(self, x):
        x = self.feat_extract(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x

import torchvision
def save_videos(batch_tensors, filename, fps=10):
    # b,samples,c,t,h,w
    n_samples = batch_tensors.shape[1]
    for idx, vid_tensor in enumerate(batch_tensors):
        video = vid_tensor.detach().cpu() 
        video = torch.clamp(video.float(), 0, 1.) 
        video = video.permute(2, 0, 1, 3, 4)  
        frame_grids = [torchvision.utils.make_grid(framesheet, nrow=int(n_samples)) for framesheet in video]  
        grid = torch.stack(frame_grids, dim=0)  
        grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1)  
        torchvision.io.write_video(filename, grid, fps=fps, video_codec='h264', options={'crf': '10'}) 

def save_mul_video(video_list, filename):
    video = torch.cat(video_list, dim=-1)
    video = video.permute(0, 2, 1, 3, 4)
    batch_variants = []
    batch_variants.append(video)
    batch_samples = torch.stack(batch_variants, dim=1)
    save_videos(batch_samples, filename, 10)  

from kornia import augmentation as K
from train_H264 import Simple3DCNN
import random

compressor = Simple3DCNN()
compressor.load_state_dict(torch.load('ckpt/model_3dcnn_crf24.pth'))

for param in compressor.parameters():
    param.requires_grad = False

attacks = {
    'identity': lambda x: x, 
    'rotation': K.RandomRotation(
        degrees=(-30, 30),
        keepdim=True,
    ),
    'gaussian_blur': K.RandomGaussianBlur(
        kernel_size=random.choice([(3,3), (5,5), (7,7)]),   
        sigma=(1,3),     
        keepdim=True,
    ),
    'gaussian_noise': K.RandomGaussianNoise(
        mean=0.,
        std=0.2,
        keepdim=True,
    ),
    'erasing': K.RandomErasing(
        keepdim=True,
    ),
}

def attack(x, h264_prob=1/6):
    B, T, C, H, W = x.shape
    x_out = []

    for b in range(B):
        clip = x[b:b+1]  # [1, T, C, H, W]

        if random.random() < h264_prob:
            with torch.no_grad():
                attacked = compressor.to(clip.device)(clip)  # [1, T, C, H, W]
            x_out.append(attacked)

        else:
            clip_frames = clip.view(T, C, H, W)  # [T, C, H, W]
            attacked_frames = []

            for t in range(T):
                attack_name = random.choice(list(attacks.keys()))
                attack_fn = attacks[attack_name]
                attacked = attack_fn(clip_frames[t:t+1])  # [1, C, H, W]
                attacked_frames.append(attacked)

            attacked_clip = torch.stack(attacked_frames, dim=0)  # [T, 1, C, H, W]
            attacked_clip = attacked_clip.permute(1, 0, 2, 3, 4)  # [1, T, C, H, W]
            x_out.append(attacked_clip)

    return torch.cat(x_out, dim=0)  # [B, T, C, H, W]










