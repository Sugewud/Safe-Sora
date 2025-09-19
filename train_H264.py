import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
from math import log10

# ===============================
#       Dataset
# ===============================

class VideoFrameDataset(Dataset):
    def __init__(self, root_raw, root_crf24, clip_ids, transform=None):
        self.root_raw = root_raw
        self.root_crf24 = root_crf24
        self.clip_ids = clip_ids
        self.transform = transform

    def __len__(self):
        return len(self.clip_ids)

    def load_clip(self, clip_root):
        frames = []
        for i in range(8):
            img_path = os.path.join(clip_root, f"frame_{i}.jpg")
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            frames.append(img)
        return torch.stack(frames, dim=0)  # [T, C, H, W]

    def __getitem__(self, idx):
        clip_id = f"{self.clip_ids[idx]:05d}"
        raw_clip = self.load_clip(os.path.join(self.root_raw, clip_id))
        crf24_clip = self.load_clip(os.path.join(self.root_crf24, clip_id))
        return raw_clip, crf24_clip  # both: [T, C, H, W]

# ===============================

# ===============================

class Simple3DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 3, kernel_size=3, padding=1),
        )

    def forward(self, x):  # x: [B, T, C, H, W]
        x = x.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
        return x

# ===============================
#       PSNR
# ===============================

def psnr(pred, target):
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return 100
    return 10 * log10(1 / mse.item())

# ===============================
#       Training Loop
# ===============================

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    transform = transforms.Compose([
        transforms.Resize((320, 512)),
        transforms.ToTensor(),
    ])

    all_ids = list(range(1000))
    train_ids = all_ids[:900]
    val_ids = all_ids[900:]

    train_dataset = VideoFrameDataset(
        root_raw="data/Panda-70M-sampled",
        root_crf24="data/Panda-70M-CRF24",
        clip_ids=train_ids,
        transform=transform
    )

    val_dataset = VideoFrameDataset(
        root_raw="data/Panda-70M-sampled",
        root_crf24="data/Panda-70M-CRF24",
        clip_ids=val_ids,
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)

    
    model = Simple3DCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    
    epochs = 10

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        for raw_clip, crf_clip in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            raw_clip, crf_clip = raw_clip.to(device), crf_clip.to(device)
            preds = model(raw_clip)
            loss = criterion(preds, crf_clip)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        
        model.eval()
        val_psnr = 0
        with torch.no_grad():
            for raw_clip, crf_clip in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                raw_clip, crf_clip = raw_clip.to(device), crf_clip.to(device)
                preds = model(raw_clip)
                val_psnr += psnr(preds, crf_clip)

        avg_val_psnr = val_psnr / len(val_loader)

        print(f"\nEpoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val PSNR = {avg_val_psnr:.2f} dB\n")

    
    torch.save(model.state_dict(), "ckpt/model_3dcnn_crf24.pth")

if __name__ == "__main__":
    train_model()