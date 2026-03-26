"""
Offline VAE feature extraction for reference-driven video training.

Usage:
    python utils/feature_extraction.py \
        --meta_file dataset_meta/meta.jsonl \
        --output_dir dataset_meta/latents \
        --output_meta dataset_meta/meta_with_latents.jsonl
"""
import argparse
import json
import os
from pathlib import Path

import torch
import torchvision.io as tvio
import torchvision.transforms.functional as TF
from tqdm import tqdm

from utils.wan_wrapper import WanVAEWrapper

# Model spatial resolution: latent 60×104 → pixel 480×832
TARGET_H, TARGET_W = 480, 832


def load_video_as_pixel(video_path: str) -> torch.Tensor:
    """Load video and return [1, C, T, H, W] float32 in [-1, 1]."""
    frames, _, _ = tvio.read_video(video_path, pts_unit="sec", output_format="TCHW")
    # frames: [T, C, H, W] uint8
    frames = frames.float() / 127.5 - 1.0
    frames = TF.resize(frames, [TARGET_H, TARGET_W], antialias=True)
    # [T, C, H, W] -> [1, C, T, H, W]
    return frames.permute(1, 0, 2, 3).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_file", required=True, help="Input JSONL meta file")
    parser.add_argument("--output_dir", required=True, help="Directory to save .pt latent files")
    parser.add_argument("--output_meta", required=True, help="Output JSONL with video_latent field added")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = WanVAEWrapper().to(device)

    with open(args.meta_file) as f:
        samples = [json.loads(l) for l in f]

    out_records = []
    for sample in tqdm(samples, desc="Encoding videos"):
        video_path = sample["video_clip"]
        stem = Path(video_path).stem
        latent_path = os.path.join(args.output_dir, f"{stem}.pt")

        if not os.path.exists(latent_path):
            pixel = load_video_as_pixel(video_path).to(device=device, dtype=torch.bfloat16)
            with torch.no_grad():
                latent = vae.encode_to_latent(pixel)  # [1, T, 16, 60, 104]
            torch.save(latent.squeeze(0).cpu(), latent_path)  # save [T, 16, 60, 104]

        out_records.append({**sample, "video_latent": latent_path})

    with open(args.output_meta, "w") as f:
        for rec in out_records:
            f.write(json.dumps(rec) + "\n")

    print(f"Done. {len(out_records)} samples written to {args.output_meta}")


if __name__ == "__main__":
    main()
