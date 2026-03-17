import h5py
import numpy as np
import shutil
import os

INPUT_FILE = "./datasets/sm_dataset_vel_1_episodes.hdf5"
OUTPUT_FILE = "./datasets/sm_dataset_vel_1_episodes_cleaned.hdf5"
MAX_FRAMES = 800

# ── Step 1: Find episodes exceeding the frame limit ──────────────────────────
print("=== Scanning for oversized episodes ===")
to_delete = []

with h5py.File(INPUT_FILE, "r") as f:
    data_group = f["data"]
    for demo_key in sorted(data_group.keys(), key=lambda x: int(x.split("_")[1])):
        demo = data_group[demo_key]
        if "actions" in demo:
            n_frames = demo["actions"].shape[0]
            flag = " ← OVERSIZED" if n_frames > MAX_FRAMES else ""
            print(f"  {demo_key}: {n_frames} frames{flag}")
            if n_frames > MAX_FRAMES:
                to_delete.append(demo_key)

print(f"\nEpisodes to delete: {to_delete}")

# ── Step 2: Copy file and delete those episodes, then renumber ────────────────
print(f"\nWriting cleaned file to: {OUTPUT_FILE}")

with h5py.File(INPUT_FILE, "r") as src, h5py.File(OUTPUT_FILE, "w") as dst:

    # Copy top-level attributes and non-data groups
    for attr_key, attr_val in src.attrs.items():
        dst.attrs[attr_key] = attr_val

    # Copy everything except the episodes we're deleting, then renumber
    src_data = src["data"]
    dst_data = dst.create_group("data")

    # Copy data group attributes (e.g. total episodes count)
    for k, v in src_data.attrs.items():
        dst_data.attrs[k] = v

    kept = [k for k in sorted(src_data.keys(), key=lambda x: int(x.split("_")[1]))
            if k not in to_delete]

    for new_idx, old_key in enumerate(kept):
        new_key = f"demo_{new_idx}"
        src_data.copy(old_key, dst_data, name=new_key)
        if old_key != new_key:
            print(f"  Renamed {old_key} → {new_key}")

    # Update the total count attribute if it exists
    if "total" in dst_data.attrs:
        dst_data.attrs["total"] = len(kept)
    if "num_demos" in dst_data.attrs:
        dst_data.attrs["num_demos"] = len(kept)

    print(f"\nKept {len(kept)} episodes (removed {len(to_delete)})")

print(f"\nDone! Cleaned file saved to: {OUTPUT_FILE}")
print("Verify it looks correct, then replace the original if needed:")
print(f"  mv {OUTPUT_FILE} {INPUT_FILE}")