import h5py
import cv2
import numpy as np
import os
import sys

def extract_images_from_h5(h5_file_path, output_dir):
    # 1. 파일 존재 여부 확인
    if not os.path.exists(h5_file_path):
        print(f"Error: 파일을 찾을 수 없습니다 -> {h5_file_path}")
        return

    # 2. 저장할 폴더 생성
    os.makedirs(output_dir, exist_ok=True)
    print(f"Reading file: {h5_file_path}")
    print(f"Saving images to: {output_dir}/")

    # 3. H5 파일 열기
    with h5py.File(h5_file_path, 'r') as f:
        # 그룹 이름(step 번호 '0', '1', '2'...)을 숫자로 정렬해서 가져오기
        # 정렬하지 않으면 0, 1, 10, 11... 순서로 뒤죽박죽 될 수 있음
        step_keys = sorted(f.keys(), key=lambda x: int(x))
        
        total_steps = len(step_keys)
        print(f"Total steps found: {total_steps}")

        for step in step_keys:
            # 해당 스텝의 그룹 접근
            group = f[step]
            
            if 'image1' not in group:
                print(f"Step {step}: 'image1' key not found, skipping.")
                continue

            # 데이터 읽기 (numpy array 형태)
            image_data = group['image1'][:]

            # --- [중요] 디코딩 과정 ---
            # 저장할 때 cv2.imencode로 압축했으므로, 읽을 때는 cv2.imdecode로 풀어야 함
            # image_data는 1D 바이트 배열임
            image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

            if image is None:
                # 만약 압축하지 않고 Raw 데이터(H,W,C)로 저장했던 파일이라면 바로 할당
                image = image_data

            # 이미지 저장 파일명 (예: image_0001.jpg)
            # zfill(6)은 000001 처럼 자릿수 맞추기 위함
            filename = f"image_{step.zfill(6)}.jpg"
            save_path = os.path.join(output_dir, filename)

            # 이미지 쓰기
            cv2.imwrite(save_path, image)

            if int(step) % 100 == 0:
                print(f"Processed step {step} / {step_keys[-1]} ...")

    print("모든 이미지 추출 완료!")

if __name__ == "__main__":
    # === 설정 부분 ===
    # 읽어올 h5 파일 경로 (여기를 본인 파일명으로 수정하세요)
    # 예: "collected_data/episode_20231223_120000.h5"
    TARGET_FILE = "episode_20251223_173119.h5" 
    
    # 이미지를 저장할 폴더 이름
    OUTPUT_FOLDER = "extracted_images"
    # ================

    extract_images_from_h5(TARGET_FILE, OUTPUT_FOLDER)