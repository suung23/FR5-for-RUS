import h5py
import numpy as np
import cv2
import os
from tqdm import tqdm

def create_model_input_video(h5_path, output_path, fps=30, target_size=224):
    """
    H5 파일의 image1을 읽어 모델 입력 전처리(Resize & Center Crop)를 적용한 후
    224x224 크기의 동영상으로 저장합니다. (GT 포인트 표시 X)
    """
    if not os.path.exists(h5_path):
        print(f"Error: File not found - {h5_path}")
        return

    try:
        with h5py.File(h5_path, 'r') as f:
            # 1. 프레임 순서 정렬
            keys = sorted([k for k in f.keys() if k.isdigit()], key=lambda x: int(x))
            
            if len(keys) == 0:
                print("No step groups found.")
                return

            print(f"Total frames: {len(keys)}")
            print(f"Output Resolution: {target_size}x{target_size} (Resize & Center Crop)")

            # 2. VideoWriter 설정
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            out = cv2.VideoWriter(output_path, fourcc, fps, (target_size, target_size))

            # 3. 프레임 처리 루프
            for k in tqdm(keys, desc="Rendering Video"):
                grp = f[k]
                
                # 이미지 로드
                img_data = grp['image1'][()]
                if img_data.size == 0:
                    continue
                
                # 원본 이미지 디코딩
                original_img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
                if original_img is None:
                    continue

                h, w = original_img.shape[:2]

                # --- [전처리] Resize Shortest Edge & Center Crop ---
                # 1. 짧은 변을 target_size(224)에 맞게 비율 유지하며 리사이즈
                scale = target_size / min(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                
                # cv2.resize 사용 (보간법은 기본인 LINEAR 혹은 CUBIC 사용 가능)
                resized_img = cv2.resize(original_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

                # 2. 중앙 좌표 계산
                start_x = (new_w - target_size) // 2
                start_y = (new_h - target_size) // 2
                
                # 3. 중앙 크롭 (실제 모델에 들어가는 이미지)
                input_frame = resized_img[start_y : start_y + target_size, 
                                          start_x : start_x + target_size]

                # 동영상 저장
                out.write(input_frame)

            out.release()
            print(f"Video saved successfully: {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # --- 설정 부분 ---
    input_h5 = "/home/medisc/FR5/ros2_ws/collected_data/suture_annotated_merge/train_filtered/episode_20260121_121927.h5" 
    output_mp4 = "model_input_only_224.mp4"
    
    # 실행
    create_model_input_video(input_h5, output_mp4, fps=30, target_size=224)