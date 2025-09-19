import os
import subprocess
from tqdm import tqdm
from pathlib import Path

SOURCE_DIR = "data/Panda-70M-sampled"
TARGET_DIR = "data/Panda-70M-CRF24"
NUM_CLIPS = 1000  
FRAMES_PER_CLIP = 8

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def simulate_crf24_one_clip(clip_id):
    src_clip = os.path.join(SOURCE_DIR, f"{clip_id:05d}")
    dst_clip = os.path.join(TARGET_DIR, f"{clip_id:05d}")
    ensure_dir(dst_clip)

    
    tmp_dir = Path("tmp_crf24")
    tmp_dir.mkdir(exist_ok=True)
    for i in range(FRAMES_PER_CLIP):
        src_frame = os.path.join(src_clip, f"frame_{i}.jpg")
        dst_frame = tmp_dir / f"frame_{i:03d}.jpg"
        os.system(f"cp '{src_frame}' '{dst_frame}'")

    
    raw_video = tmp_dir / "raw.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-framerate", "30", "-i", str(tmp_dir / "frame_%03d.jpg"),
        "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv420p", str(raw_video)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    
    compressed_video = tmp_dir / "compressed.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-i", str(raw_video),
        "-c:v", "libx264", "-crf", "24", "-preset", "slow", str(compressed_video)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    
    subprocess.run([
        "ffmpeg", "-y", "-i", str(compressed_video), str(tmp_dir / "frame_%03d.jpg")
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    
    for i in range(FRAMES_PER_CLIP):
        compressed_frame = tmp_dir / f"frame_{i:03d}.jpg"
        dst_frame = os.path.join(dst_clip, f"frame_{i}.jpg")
        os.system(f"cp '{compressed_frame}' '{dst_frame}'")

    
    for f in tmp_dir.glob("frame_*.jpg"):
        f.unlink()
    for f in tmp_dir.glob("*.mp4"):
        f.unlink()

def main():
    ensure_dir(TARGET_DIR)
    for i in tqdm(range(NUM_CLIPS), desc="Simulating CRF=24"):
        simulate_crf24_one_clip(i)

if __name__ == "__main__":
    main()