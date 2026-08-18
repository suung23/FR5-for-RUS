import h5py
import numpy as np
import pathlib
import os
import shutil
from tqdm import tqdm

def preprocess_episode(episode):
    # 1. Clean Gripper State (Fix jumps and normalize)
    gripper_state = episode['gripper_state']
    desired_gripper_state = episode['action'][..., -1:]
    
    # Identify invalid values (outliers > 105)
    is_outlier = gripper_state > 105.0
    
    if np.any(is_outlier):
        # Convert to float for manipulation
        gripper_clean = gripper_state.copy().astype(np.float32)
        gripper_clean[is_outlier] = np.nan
        
        # Interpolate
        # Flatten for interpolation
        flat_clean = gripper_clean.flatten()
        nans, x = np.isnan(flat_clean), lambda z: z.nonzero()[0]
        flat_clean[nans] = np.interp(x(nans), x(~nans), flat_clean[~nans])
        episode['gripper_state'] = flat_clean.reshape(gripper_state.shape)

    # 2. Trim Initialization
    twist = episode['action'][..., :-1]
    twist_mag = np.linalg.norm(twist, axis=-1)
    
    move_threshold = 1e-3 
    is_moving = twist_mag > move_threshold
    
    if np.any(is_moving):
        start_idx = np.argmax(is_moving)
    else:
        print("Warning: No movement found")
        return None

    # 3. Trim End
    desired = desired_gripper_state.flatten()
    end_idx = len(desired)
    
    for i in range(len(desired) - 1, start_idx, -1):
        if desired[i] > 50.0:
            end_idx = i + 1
            break
    
    if start_idx >= end_idx:
        print(f"Warning: Empty episode after trim {start_idx} {end_idx}")
        return None

    # Slice all arrays
    for k, v in episode.items():
        episode[k] = v[start_idx:end_idx]
        
    return episode

def main():
    source_dir = pathlib.Path("/home/rosota/FR5/ros2_ws/collected_data/peg_transfer")
    target_dir = pathlib.Path("/home/rosota/FR5/ros2_ws/collected_data/peg_transfer_clean")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    episodes = [
        "episode_20251223_170336_p_d_yc",
        "episode_20251223_170528_p_u_yc",
        "episode_20251223_170806_p_d_yc",
        "episode_20251223_170941_p_u_yc",
        "episode_20251223_171237_p_u_yc",
        "episode_20251223_171356_p_d_yc",
        "episode_20251223_171507_p_u_yc",
        "episode_20251223_171608_p_d_yc",
        "episode_20251223_171730_p_u_yc",
        "episode_20251223_171843_p_d_yc",
        "episode_20251223_172002_p_u_yc",
        "episode_20251223_172107_p_d_yc",
        "episode_20251223_172242_p_u_yc",
        "episode_20251223_172357_p_d_yc",
        "episode_20251223_172457_p_u_yc",
        "episode_20251223_172555_p_d_yc",
        "episode_20251223_172658_p_u_yc",
        "episode_20251223_172818_p_d_yc",
        "episode_20251223_172916_p_u_yc",
        "episode_20251223_174421_f_d_ldh",
        "episode_20251223_174606_f_u_ldh",
        "episode_20251223_180224_p_d_yc",
        "episode_20251223_200754_f_u_yc",
        "episode_20251223_201058_p_d_yc",
        "episode_20251223_201758_p_d_yc",
        "episode_20251223_202040_p_d_yc",
        "episode_20251223_203103_p_u_yc",
        "episode_20251223_203208_p_d_yc",
        "episode_20251223_203303_p_u_yc",
        "episode_20251223_203355_p_d_yc",
        "episode_20251223_203446_p_u_yc",
        "episode_20251223_203536_p_d_yc",
        "episode_20251223_234656_p_u",
        "episode_20251223_234800_p_d",
        "episode_20251223_234924_p_u"
    ]

    print(f"Start processing {len(episodes)} episodes...")
    
    for ep_name in tqdm(episodes):
        src_path = source_dir / f"{ep_name}.h5"
        dst_path = target_dir / f"{ep_name}.h5"
        
        if not src_path.exists():
            print(f"Skipping {ep_name}, not found.")
            continue
            
        with h5py.File(src_path, 'r') as f_in:
            keys = sorted(list(f_in.keys()), key=lambda x: int(x))
            
            # Additional keys that might exist
            # Let's inspect the first key to know all keys
            first_grp = f_in[keys[0]]
            all_keys = list(first_grp.keys())
            
            # Re-initialize with all keys
            episode_data = {k: [] for k in all_keys}
            
            # Helper for action construction (for trimming logic)
            actions = []
            
            for k in keys:
                grp = f_in[k]
                for dkey in all_keys:
                    episode_data[dkey].append(grp[dkey][()])
                
                # Construct action for logic
                twist = grp['fr5_right_desired_twist'][()]
                grip_des = grp['fr5_right_desired_gripper_state'][()]
                actions.append(np.concatenate([twist, grip_des]))
            
            # Convert to numpy
            for dkey in all_keys:
                episode_data[dkey] = np.array(episode_data[dkey])
            
            # Make the 'episode' dict for preprocess function
            # The function expects 'gripper_state', 'action', 'relative_ee_pose'
            # We map our keys to that
            processing_dict = {
                'gripper_state': episode_data['fr5_right_gripper_state'],
                'action': np.array(actions),
                # pass other items as well so they get sliced
            }
            # Add all other keys to processing_dict to ensure they get sliced
            for dkey in all_keys:
                if dkey == 'fr5_right_gripper_state': continue
                processing_dict[dkey] = episode_data[dkey]
                
            # Process
            processed = preprocess_episode(processing_dict)
            
            if processed is None:
                print(f"Episode {ep_name} removed during processing.")
                continue
                
            # Save to new H5
            with h5py.File(dst_path, 'w') as f_out:
                length = len(processed['gripper_state'])
                for i in range(length):
                    grp = f_out.create_group(str(i))
                    for dkey in all_keys:
                        # If key was 'fr5_right_gripper_state', it was modified in place in processed['gripper_state']
                        if dkey == 'fr5_right_gripper_state':
                            data = processed['gripper_state'][i]
                        else:
                            data = processed[dkey][i]
                        
                        grp.create_dataset(dkey, data=data)
                        
    print("Processing complete.")

if __name__ == "__main__":
    main()
