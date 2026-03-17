import h5py
import numpy as np

with h5py.File("./datasets/sm_dataset_vel_65_episodes_cleaned.hdf5", "r") as f:
    def print_item(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"  [Dataset] {name}: shape={obj.shape}, dtype={obj.dtype}")
        else:
            print(f"  [Group]   {name}/")
    
    print("=== File Structure ===")
    f.visititems(print_item)
    
    print("\n=== Episodes ===")
    data_group = f["data"]

    # Print top-level attributes
    if data_group.attrs:
        print("\n  Attributes:")
        for k, v in data_group.attrs.items():
            print(f"    {k}: {v}")

    # Iterate over demo_0, demo_1, ...
    for demo_key in sorted(data_group.keys(), key=lambda x: int(x.split("_")[1])):
        demo = data_group[demo_key]
        print(f"\n--- {demo_key} ---")

        if "observation" in demo:
            obs_group = demo["observation"]

            if "state" in obs_group:
                state = obs_group["state"][:]
                print(f"  obs/state:  shape={state.shape}, dtype={state.dtype}, min={state.min():.3f}, max={state.max():.3f}")

            if "images" in obs_group:
                wrist_images = obs_group["images"]["wrist"]
                print(f"  obs/images: shape={wrist_images.shape}, dtype={wrist_images.dtype}")
                base_images = obs_group["images"]["base"]
                print(f"  obs/images: shape={base_images.shape}, dtype={base_images.dtype}")

        if "actions" in demo:
            actions = demo["actions"][:]
            print(f"  actions:    shape={actions.shape}, dtype={actions.dtype}, min={actions.min():.3f}, max={actions.max():.3f}")