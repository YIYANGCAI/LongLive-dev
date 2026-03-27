"""
Offline VAE feature extraction for reference-driven video training.

Usage:
    python utils/feature_extraction.py \
        --meta_file dataset_meta/meta.jsonl \
        --output_dir dataset_meta/latents
"""
import argparse
import json
import os
from pathlib import Path

import torch
import torchvision.io as tvio
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from utils.wan_wrapper import WanVAEWrapper


def load_video(video_path: str, target_h: int, target_w: int, num_frames: int):
    """Load video, resize, sample/trim to num_frames. Returns [1, C, T, H, W] float32 in [-1,1] or None if too short."""
    frames, _, _ = tvio.read_video(video_path, pts_unit="sec", output_format="TCHW")
    # frames: [T, C, H, W] uint8
    if frames.shape[0] < num_frames:
        return None
    frames = frames[:num_frames]
    frames = frames.float() / 127.5 - 1.0
    frames = TF.resize(frames, [target_h, target_w], antialias=True)
    # [T, C, H, W] -> [1, C, T, H, W]
    return frames.permute(1, 0, 2, 3).unsqueeze(0)


def load_ref_image(image_path: str, target_h: int, target_w: int) -> torch.Tensor:
    """Load reference image, pad to target aspect ratio with white, resize. Returns [1, C, 1, H, W] in [-1,1]."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    target_ratio = target_w / target_h
    src_ratio = w / h

    if src_ratio > target_ratio:
        # wider than target: pad height
        new_h = round(w / target_ratio)
        pad_top = (new_h - h) // 2
        pad_bottom = new_h - h - pad_top
        padded = Image.new("RGB", (w, new_h), (255, 255, 255))
        padded.paste(img, (0, pad_top))
    else:
        # taller than target: pad width
        new_w = round(h * target_ratio)
        pad_left = (new_w - w) // 2
        pad_right = new_w - w - pad_left
        padded = Image.new("RGB", (new_w, h), (255, 255, 255))
        padded.paste(img, (pad_left, 0))

    padded = padded.resize((target_w, target_h), Image.LANCZOS)
    t = torch.from_numpy(__import__('numpy').array(padded)).permute(2, 0, 1).float() / 127.5 - 1.0
    # [C, H, W] -> [1, C, 1, H, W]
    return t.unsqueeze(0).unsqueeze(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_file", default="dataset_meta/meta.jsonl")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--video_width", type=int, default=832)
    parser.add_argument("--video_height", type=int, default=480)
    parser.add_argument("--video_frame", type=int, default=81)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = WanVAEWrapper().to(device)

    with open(args.meta_file) as f:
        samples = [json.loads(l) for l in f]

    skipped = 0
    for sample in tqdm(samples, desc="Encoding"):
        video_path = sample["video_clip"]
        ref_paths = sample["reference_image"]  # list of paths
        prompt = sample["prompt"]

        stem = Path(video_path).stem
        out_path = os.path.join(args.output_dir, f"{stem}.pt")
        if os.path.exists(out_path):
            continue

        # Load video
        pixel = load_video(video_path, args.video_height, args.video_width, args.video_frame)
        if pixel is None:
            print(f"[SKIP] {video_path}: fewer than {args.video_frame} frames")
            skipped += 1
            continue

        # Encode video latent: [1, C, T, H, W] -> [1, T, 16, h, w]
        pixel = pixel.to(device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            latent = vae.encode_to_latent(pixel)  # [1, T, 16, h, w]

        # Encode reference images: each [1, C, 1, H, W] -> [1, 1, 16, h, w]
        ref_latents = []
        for rp in ref_paths:
            ref_pixel = load_ref_image(rp, args.video_height, args.video_width)
            ref_pixel = ref_pixel.to(device=device, dtype=torch.bfloat16)
            with torch.no_grad():
                ref_lat = vae.encode_to_latent(ref_pixel)  # [1, 1, 16, h, w]
            ref_latents.append(ref_lat)
        # [1, num_refs, 16, h, w]
        latents_ref = torch.cat(ref_latents, dim=1)

        torch.save({
            "latent": latent.cpu(),          # [1, T, 16, h, w]
            "latents_ref": latents_ref.cpu(), # [1, num_refs, 16, h, w]
            "prompt": prompt,
        }, out_path)

    print(f"Done. Skipped {skipped} samples.")


if __name__ == "__main__":
    main()
