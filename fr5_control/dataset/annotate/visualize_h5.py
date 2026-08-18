import h5py
import numpy as np
import cv2
import argparse
import os
from tqdm import tqdm

def create_video_from_h5(h5_path, output_path, fps=30):
    if not os.path.exists(h5_path):
        print(f"Error: File not found - {h5_path}")
        return

    try:
        with h5py.File(h5_path, 'r') as f:
            # 1. 그룹(Step) 이름 정렬 ('0', '1', '2' ... 순서)
            # 숫자가 아닌 키(metadata 등)는 제외
            keys = sorted([k for k in f.keys() if k.isdigit()], key=lambda x: int(x))
            
            if len(keys) == 0:
                print("No step groups found in H5 file.")
                return

            print(f"Total frames found: {len(keys)}")

            # 2. 첫 번째 프레임을 읽어 비디오 크기 결정
            first_grp = f[keys[0]]
            first_img_data = first_grp['image1'][()]
            first_img = cv2.imdecode(np.frombuffer(first_img_data, np.uint8), cv2.IMREAD_COLOR)
            
            if first_img is None:
                print("Failed to decode the first frame.")
                return

            height, width = first_img.shape[:2]
            print(f"Video Resolution: {width}x{height}")

            # 3. VideoWriter 설정
            # 코덱: mp4v (macOS/Windows/Linux 호환성 좋음)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            # 4. 프레임 순회하며 그리기
            for k in tqdm(keys, desc="Rendering Video"):
                grp = f[k]
                
                # 이미지 디코딩
                img_data = grp['image1'][()]
                if img_data.size == 0:
                    continue
                
                frame = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                # GT Sparse Depth 로드
                if 'gt_sparse_depths' in grp:
                    gt_points = grp['gt_sparse_depths'][()]
                    # 모양 확인 및 Reshape (N, 3)
                    if gt_points.ndim == 1:
                        gt_points = gt_points.reshape(-1, 3)
                    
                    # 시각화 (점 그리기)
                    for point in gt_points:
                        if len(point) < 3: continue
                        
                        u, v, d = point[0], point[1], point[2]
                        x, y = int(u), int(v)

                        # 화면 밖 좌표 예외처리
                        if x < 0 or x >= width or y < 0 or y >= height:
                            continue

                        # 점 그리기 (초록색 원)
                        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1) # 내부 채움
                        cv2.circle(frame, (x, y), 5, (255, 255, 255), 1) # 흰색 테두리

                        # 텍스트 그리기 (Depth 값)
                        text = f"{d:.2f}m"
                        
                        # 텍스트 배경 (검은색 테두리 효과)
                        cv2.putText(frame, text, (x + 8, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
                        # 텍스트 본문 (노란색)
                        cv2.putText(frame, text, (x + 8, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

                # 프레임 쓰기
                out.write(frame)

            # 5. 종료 및 저장
            out.release()
            print(f"Video saved successfully: {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # --- 설정 부분 ---
    # 확인하고 싶은 H5 파일 경로
    input_h5 = "/home/medisc/FR5/ros2_ws/collected_data/suture_new_annotated/episode_20260206_182112.h5" 
    
    # 저장될 동영상 경로
    output_mp4 = "visualization_result.mp4"
    
    # 실행
    # (폴더 내의 가장 최근 파일을 자동으로 찾고 싶다면 아래 주석 해제)
    # import glob
    # list_of_files = glob.glob('collected_data/*.h5') 
    # if list_of_files:
    #     input_h5 = max(list_of_files, key=os.path.getctime) # 가장 최근 파일
    
    print(f"Input: {input_h5}")
    create_video_from_h5(input_h5, output_mp4, fps=30)