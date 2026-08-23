import h5py
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Visualize HDF5 Dataset (Video + Plots)")
    parser.add_argument('--file', type=str, required=True, help='Path to .h5 file')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save output (default: ./results/<filename>/)')
    args = parser.parse_args()

    h5_path = os.path.abspath(args.file)
    if not os.path.exists(h5_path):
        print(f"Error: File not found: {h5_path}")
        sys.exit(1)

    filename = os.path.splitext(os.path.basename(h5_path))[0]
    
    # Determine Output Directory
    if args.output_dir is None:
        # Default: ./results/<filename> inside the script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "results", filename)
    else:
        output_dir = os.path.join(args.output_dir, filename)

    os.makedirs(output_dir, exist_ok=True)
    print(f"Processing: {filename}")
    print(f"Output Directory: {output_dir}")

    # Data containers
    data_history = {}
    timestamps = []
    images = []

    with h5py.File(h5_path, 'r') as f:
        # Sort keys numerically to ensure time order (0, 1, 2...)
        keys = sorted(f.keys(), key=lambda k: int(k))
        
        print(f"Found {len(keys)} steps.")

        # Initialize data structure based on first step
        if len(keys) > 0:
            first_step = f[keys[0]]
            for key in first_step.keys():
                if key == 'image1':
                    continue
                data_history[key] = []

        for k in keys:
            group = f[k]
            timestamps.append(group['timestamp'][()])
            
            # Decode Image
            img_data = np.array(group['image1'])
            # Check if it's already decoded or compressed
            if img_data.ndim == 1:
                # Compressed (uint8 array)
                decoded = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                if decoded is not None:
                    images.append(decoded)
                else:
                    print(f"Warning: Failed to decode image at step {k}")
            else:
                # Raw (H, W, 3)
                images.append(img_data)

            # Collect other data
            for key in data_history.keys():
                val = np.array(group[key])
                data_history[key].append(val)

    # 1. Generate Video
    if images:
        video_path = os.path.join(output_dir, f"{filename}_video.webm")
        height, width, layers = images[0].shape
        # MP4V codec
        fourcc = cv2.VideoWriter_fourcc(*'vp80')
        out = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))

        for img in images:
            out.write(img)
        out.release()
        print(f"Video saved to: {video_path}")
    else:
        print("No images found to generate video.")


    # 2. Generate Plots
    if not data_history:
        print("No data found to plot.")
        return

    # Convert lists to numpy arrays
    for key in data_history:
        data_history[key] = np.array(data_history[key])
    timestamps = np.array(timestamps)
    # Relative time
    if len(timestamps) > 0:
        rel_time = timestamps - timestamps[0]
    else:
        rel_time = []

    # Identify prefixes (arms)
    prefixes = set()
    for key in data_history.keys():
        if key.startswith('fr5_left'):
            prefixes.add('fr5_left')
        elif key.startswith('fr5_right'):
            prefixes.add('fr5_right')
    
    # If no prefixes found (unexpected), just plot everything simply
    if not prefixes:
        prefixes.add('')

    for prefix in prefixes:
        # Check if this prefix actually has data
        keys_for_prefix = [k for k in data_history.keys() if (prefix in k) or (prefix == '')]
        if not keys_for_prefix:
            continue
            
        print(f"Plotting data for: {prefix}")
        
        # Create a figure with subplots
        fig, axs = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
        fig.suptitle(f"Robot State: {prefix} - {filename}", fontsize=16)
        
        # Helper to find key
        def get_key(base):
            k = f"{prefix}_{base}" if prefix else base
            # Handle potential underscore differences if manual keys existed
            if k not in data_history and k.replace('_', '') in data_history:
                return k.replace('_', '')
            return k

        # 1. Gripper
        ax = axs[0, 0]
        k_curr = get_key("gripper_state")
        k_des = get_key("desired_gripper_state")
        if k_curr in data_history:
            ax.plot(rel_time, data_history[k_curr], label='Current')
        if k_des in data_history:
            ax.plot(rel_time, data_history[k_des], label='Desired', linestyle='--')
        ax.set_title("Gripper State")
        ax.set_ylabel("Width/State")
        ax.legend()
        ax.grid(True)

        # 2. EE Position
        ax = axs[0, 1]
        k = get_key("relative_ee_pose")
        if k in data_history:
            # Shape (N, 7) -> Pos(3) + Quat(4)
            pos = data_history[k][:, :3]
            ax.plot(rel_time, pos[:, 0], label='X')
            ax.plot(rel_time, pos[:, 1], label='Y')
            ax.plot(rel_time, pos[:, 2], label='Z')
            ax.set_title("EE Position (Relative)")
            ax.set_ylabel("Position (m)")
            ax.legend()
            ax.grid(True)

        # 3. EE Orientation (Quaternion)
        ax = axs[1, 1]
        if k in data_history:
            quat = data_history[k][:, 3:]
            ax.plot(rel_time, quat[:, 0], label='qx')
            ax.plot(rel_time, quat[:, 1], label='qy')
            ax.plot(rel_time, quat[:, 2], label='qz')
            ax.plot(rel_time, quat[:, 3], label='qw')
            ax.set_title("EE Orientation (Quaternion)")
            ax.set_ylabel("Quaternion")
            ax.legend()
            ax.grid(True)
        
        # 4. Twist (Linear)
        ax = axs[1, 0]
        k = get_key("desired_twist")
        if k in data_history:
            # Shape (N, 6) -> Lin(3) + Ang(3)
            lin = data_history[k][:, :3]
            ax.plot(rel_time, lin[:, 0], label='vx')
            ax.plot(rel_time, lin[:, 1], label='vy')
            ax.plot(rel_time, lin[:, 2], label='vz')
            ax.set_title("Desired Twist (Linear)")
            ax.set_ylabel("Vel (m/s)")
            ax.legend()
            ax.grid(True)

        # 5. Twist (Angular)
        ax = axs[2, 0]
        if k in data_history:
            ang = data_history[k][:, 3:]
            ax.plot(rel_time, ang[:, 0], label='wx')
            ax.plot(rel_time, ang[:, 1], label='wy')
            ax.plot(rel_time, ang[:, 2], label='wz')
            ax.set_title("Desired Twist (Angular)")
            ax.set_ylabel("Vel (rad/s)")
            ax.legend()
            ax.grid(True)

        # 6. Joint States
        ax = axs[2, 1]
        k = get_key("joint_states")
        if k in data_history:
            joints = data_history[k]
            # Assumes 6 joints usually
            for i in range(joints.shape[1]):
                ax.plot(rel_time, joints[:, i], label=f'J{i+1}')
            ax.set_title("Joint States")
            ax.set_ylabel("Position (rad)")
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
            ax.grid(True)

        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"{filename}_{prefix}_plot.png")
        plt.savefig(plot_path)
        print(f"Plot saved to: {plot_path}")
        plt.close()

if __name__ == "__main__":
    main()
