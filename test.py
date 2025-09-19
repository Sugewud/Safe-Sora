
import torch
import os
from tqdm import tqdm
import numpy as np
import numpy as np
import torch
import os
from torchvision.utils import save_image
from custom_modules import EmbeddingNet, RevealNet, save_mul_video
from Adaptive_Embedding import Adaptive_Embedding, revert_order
from omegaconf import OmegaConf
import torchvision.transforms as T
from PIL import Image

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('>>> Model checkpoint loading')
vae = torch.load("ckpt/vae.pth", map_location="cuda:0")
print('>>> Finish!')
vae.eval()
vae.to(device)

model_dir = os.path.join('ckpt/model_latest.pth')
config = OmegaConf.load("configs/inference_t2v_512_v2.0.yaml")
ddconfig = config.model.params.first_stage_config.params.ddconfig

model = torch.load(model_dir, map_location="cuda:0")
wm_embedder = EmbeddingNet(1, vae.decoder.state_dict(), **ddconfig)
wm_embedder.load_state_dict(model["wm_embedder"])
wm_embedder = wm_embedder.to(device)

wm_extractor = RevealNet(batch_size = 1)
wm_extractor.load_state_dict(model["wm_extractor"])
wm_extractor = wm_extractor.to(device)

adaptive_embedding = Adaptive_Embedding()
adaptive_embedding.load_state_dict(model["adaptive_embedding"])
adaptive_embedding = adaptive_embedding.to(device)

wm_embedder.eval()
wm_extractor.eval()
adaptive_embedding.eval()

def Eval(video_latent, watermark_path):
    image_path = watermark_path
    transform = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])
    image_data = Image.open(image_path).convert('RGB')
    image_data = transform(image_data)  
    os.makedirs("results", exist_ok=True)
    
    with torch.no_grad():
        video_latent = video_latent.cuda() # [1, 1, 4, 16, 40, 64]
        secret = image_data.unsqueeze(0).cuda()
        
        video_latent = 1. / 0.18215 * video_latent
        video_latent = video_latent.permute(0, 2, 1, 3, 4)  
        
        decoded_frames = [vae.decode(frame) for frame in torch.unbind(video_latent, dim=1)]
        video_data = torch.stack(decoded_frames, dim=1)  # [1, 16, 3, 312, 520]

        ################## forward ####################
        patches = secret.unfold(2, 16, 16).unfold(3, 16, 16)  # [batch_size, 3, 16, 16, 16, 16]
        patches = patches.contiguous().view(1, 3, 256, 16, 16) # [batch_size, 3, 256, 16, 16]
        
        position_encoding = torch.arange(256).unsqueeze(0).repeat(1, 1) 
        position_binary = position_encoding.unsqueeze(-1).bitwise_and(1 << torch.arange(8)).ne(0).long() 
        position_binary = torch.where(position_binary == 0, torch.tensor(-1, device=position_binary.device), position_binary) 
        position_channel = position_binary.unsqueeze(-1).expand(-1, -1, -1, 32).to(device) 
        position_channel = position_channel.reshape(1, 1, 256, 16, 16).to(device) 
        patches = torch.cat([patches, position_channel.float()], dim=1) 
        original_patches = patches

        reconst_video_w_list, watermark_list = [], []
        for video_latent_cur in [video_latent[:, :8, :, :, :], video_latent[:, 8:, :, :, :]]:
            patches, _ = adaptive_embedding(original_patches, video_latent_cur)

            patches = patches.view(1, 4, 8, 4, 8, 16, 16)
            patches = patches.permute(0, 1, 2, 3, 5, 4, 6).contiguous()
            patches = patches.view(1, 4, 8, 64, 128).permute(0, 2, 1, 3, 4)

            secret_patch = patches.reshape(-1, *patches.shape[2:])
            cover = video_latent_cur.reshape(-1, *video_latent_cur.shape[2:])  # [batch_size * num_frames, C, H, W]
            cover = vae.post_quant_conv(cover)  # [batch_size * num_frames, C, H, W]

            stego_patch = wm_embedder(cover, secret_patch)  # [batch_size * num_frames, C, H, W]
            reconst_video_w = stego_patch.reshape(1, 8, *stego_patch.shape[1:])
            reconst_video_w_list.append(reconst_video_w)
            
            watermark_exact = wm_extractor(reconst_video_w)  

            watermark_exact = watermark_exact.permute(0, 2, 1, 3, 4)
            watermark_exact = watermark_exact.view(1, 4, 8, 4, 16, 8, 16)
            watermark_exact = watermark_exact.permute(0, 1, 2, 3, 5, 4, 6).contiguous()
            watermark_exact = watermark_exact.view(1, 4, 256, 16, 16)

            watermark, position_channel = watermark_exact[:, :3, :, :, :], watermark_exact[:, 3:, :, :, :]
            
            B = position_channel.size(0)                                    
            
            bit_prob = ((position_channel + 1) / 2.0)                       # [B,256*8,H,W] → [0,1]
            bit_prob = bit_prob.view(B, 256, 8, -1).mean(-1)                # [B,256,8]
            conf = (bit_prob - 0.5).abs().mean(-1)                          # [B,256]
            bit_bin = (bit_prob > 0.5).long()                               # [B,256,8]
            weights = (1 << torch.arange(8, device=bit_bin.device)).long()  # [8]
            position_encoding = (bit_bin * weights).sum(-1)                 # [B,256] ∈ [0,255]
            indices     = position_encoding.view(-1)                        # [(B*256)]
            conf_flat   = conf.view(-1)                                     # [(B*256)]

            index_map   = torch.full((B, 256), -1,  dtype=torch.long,
                                    device=indices.device)                 
            best_conf   = torch.full((B, 256), -1., device=indices.device)  

            for global_idx, (pos, con) in enumerate(zip(indices, conf_flat)):
                b = global_idx // 256           
                if con > best_conf[b, pos]:       
                    best_conf[b, pos] = con
                    index_map[b, pos] = global_idx

            for b in range(B):
                all_patch_id    = torch.arange(b*256, (b+1)*256, device=indices.device)
                kept_patch_id   = index_map[b][index_map[b] != -1]
                lost_patch_id   = torch.tensor(list(set(all_patch_id.tolist()) -
                                                    set(kept_patch_id.tolist())),
                                            device=indices.device, dtype=torch.long)
                if lost_patch_id.numel() == 0:
                    continue

                vacant_pos      = (index_map[b] == -1).nonzero(as_tuple=False).squeeze(1)
                pred_pos_lost   = indices[lost_patch_id]                     
                
                dist            = (pred_pos_lost.unsqueeze(1) - vacant_pos.unsqueeze(0)).abs()
                assign          = dist.argmin(dim=1)                         

                for lp, vp_idx in zip(lost_patch_id, assign):
                    vpos = vacant_pos[vp_idx]
                    if index_map[b, vpos] == -1:                             
                        index_map[b, vpos] = lp

            
            B, C, N, H, W = watermark_exact.size()  
            out_x_2d = watermark_exact.permute(0, 2, 1, 3, 4).contiguous().view(B, N, -1)  # [B,256,4*16*16]
            original_order_out = revert_order(out_x_2d, index_map)
            watermark_and_pos = original_order_out.view(B, N, C, H, W).permute(0,2,1,3,4).contiguous()
            watermark = watermark_and_pos[:, :3, :, :, :]
            
            watermark = watermark.reshape(1, 3, 16, 16, 16, 16) 
            watermark = watermark.permute(0, 1, 2, 4, 3, 5).contiguous().view(1, 3, 256, 256)

            watermark_list.append(watermark)
        
        secret_rev = (watermark_list[0] + watermark_list[1]) / 2.0
        stego_list = torch.cat((reconst_video_w_list[0], reconst_video_w_list[1]), dim = 1)
        cover_list = video_data

        cover_list = (torch.clamp(cover_list.float(), -1., 1.) + 1.0) / 2.0
        stego_list = (torch.clamp(stego_list.float(), -1., 1.) + 1.0) / 2.0
        secret = (torch.clamp(secret.float(), -1., 1.) + 1.0) / 2.0
        secret_rev = (torch.clamp(secret_rev.float(), -1., 1.) + 1.0) / 2.0
        combined_img = torch.cat([secret, secret_rev, abs(secret - secret_rev) * 5], dim=-1) 

        save_image(combined_img, os.path.join('results', 'secret_combined.png'))
        print("[✓] Image saved to: results/secret_combined.png")
        save_mul_video([cover_list, stego_list, abs(cover_list - stego_list) * 5], os.path.join('results', 'cover_combined.mp4'))
        print("[✓] Video saved to: results/cover_combined.mp4")