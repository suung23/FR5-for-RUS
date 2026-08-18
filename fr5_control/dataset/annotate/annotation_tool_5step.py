import os
# 리눅스/NAS 환경 호환성 (File Locking 해제)
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import tkinter as tk
from tkinter import filedialog, messagebox
import h5py
import numpy as np
import cv2
import glob
import json
import shutil
from PIL import Image, ImageTk

class SurgicalAnnotationTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Robotic Surgery Annotator V8 (5-Task Segmentation)")
        self.root.geometry("1600x1100") 

        # --- Settings ---
        self.view_scale = 2.0  

        # --- Data Variables ---
        self.data_dir = ""
        self.file_list = []
        self.current_h5 = None
        self.current_file_path = ""
        
        # [변경] 이미지를 미리 다 읽지 않고, 키(Key) 리스트만 저장합니다.
        self.step_keys = [] 
        self.total_frames = 0
        self.current_frame_idx = 0
        
        # --- Annotation Variables ---
        self.insertion_pt = None
        self.exit_pt = None
        
        # [변경] Task Splits: 2개 → 4개 (5단계 분할)
        self.split_idx_1 = 0  # End of Needle_Pickup
        self.split_idx_2 = 0  # End of Needle_Handoff
        self.split_idx_3 = 0  # End of Needle_Insertion
        self.split_idx_4 = 0  # End of Needle_Extraction
        
        # Crop Variables
        self.crop_start = 0
        self.crop_end = 0
        
        self.click_mode = 'none'

        # --- Playback Variables ---
        self.is_playing = False
        self.play_speed_ms = 33 # 약 30fps

        self._init_ui()

    def _init_ui(self):
        main_paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashwidth=5, bg="#cdcdcd")
        main_paned.pack(fill=tk.BOTH, expand=True)

        # --- [Left] Sidebar ---
        sidebar_frame = tk.Frame(main_paned, width=400, bg="#f0f0f0")
        main_paned.add(sidebar_frame)
        
        tk.Label(sidebar_frame, text="File List", bg="#ddd", font=("Arial", 14, "bold")).pack(fill=tk.X, pady=2)
        
        tk.Button(sidebar_frame, text="Open Folder", command=self.load_folder, bg="#e1e1e1", font=("Arial", 11)).pack(fill=tk.X, padx=5, pady=5)
        
        list_scroll = tk.Scrollbar(sidebar_frame)
        list_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.file_listbox = tk.Listbox(sidebar_frame, selectmode=tk.SINGLE, yscrollcommand=list_scroll.set, font=("Consolas", 14))
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        list_scroll.config(command=self.file_listbox.yview)
        self.file_listbox.bind('<<ListboxSelect>>', self.on_file_select)

        # --- [Right] Work Area ---
        work_frame = tk.Frame(main_paned, bg="white")
        main_paned.add(work_frame)

        # Top Bar
        top_bar = tk.Frame(work_frame, bd=1, relief=tk.RAISED)
        top_bar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        self.lbl_status = tk.Label(top_bar, text="No file selected", font=("Arial", 12, "bold"))
        self.lbl_status.pack(side=tk.LEFT, padx=10)
        
        tk.Button(top_bar, text="SAVE (Smart Save)", command=self.save_data, bg="#ffdddd", fg="red", font=("Arial", 16, "bold")).pack(side=tk.RIGHT, padx=10, pady=5)

        # Content
        content_box = tk.Frame(work_frame)
        content_box.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.canvas_frame = tk.Frame(content_box, bd=2, relief=tk.SUNKEN)
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg="black", cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        # Controls (with Scrollbar for more buttons)
        ctrl_container = tk.Frame(content_box, width=320)
        ctrl_container.pack(side=tk.RIGHT, fill=tk.Y, padx=10)
        
        ctrl_canvas = tk.Canvas(ctrl_container, width=300)
        ctrl_scrollbar = tk.Scrollbar(ctrl_container, orient="vertical", command=ctrl_canvas.yview)
        ctrl_frame = tk.Frame(ctrl_canvas)
        
        ctrl_frame.bind("<Configure>", lambda e: ctrl_canvas.configure(scrollregion=ctrl_canvas.bbox("all")))
        ctrl_canvas.create_window((0, 0), window=ctrl_frame, anchor="nw")
        ctrl_canvas.configure(yscrollcommand=ctrl_scrollbar.set)
        
        ctrl_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ctrl_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # [1] Needle Location
        tk.Label(ctrl_frame, text="[1] Needle Location", font=("Arial", 11, "bold")).pack(pady=10)
        self.btn_ins = tk.Button(ctrl_frame, text="Set Insertion (Green)", command=lambda: self.set_mode('insertion'), width=25)
        self.btn_ins.pack(pady=2)
        self.lbl_ins = tk.Label(ctrl_frame, text="Insertion: None", fg="green", font=("Arial", 9))
        self.lbl_ins.pack(pady=(0, 10))

        self.btn_exit = tk.Button(ctrl_frame, text="Set Exit (Red)", command=lambda: self.set_mode('exit'), width=25)
        self.btn_exit.pack(pady=2)
        self.lbl_exit = tk.Label(ctrl_frame, text="Exit: None", fg="red", font=("Arial", 9))
        self.lbl_exit.pack(pady=(0, 10))
        
        tk.Frame(ctrl_frame, height=2, bd=1, relief=tk.SUNKEN).pack(fill=tk.X, pady=10)

        # [2] Crop Range
        tk.Label(ctrl_frame, text="[2] Crop Range", font=("Arial", 11, "bold")).pack(pady=10)
        
        btn_frame = tk.Frame(ctrl_frame)
        btn_frame.pack()
        tk.Button(btn_frame, text="Set Start (Front Cut)", command=self.set_crop_start, bg="#ddd").pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="Set End (Back Cut)", command=self.set_crop_end, bg="#ddd").pack(side=tk.LEFT, padx=2)
        
        self.lbl_crop = tk.Label(ctrl_frame, text="Valid Range: 0 ~ 0", fg="black", font=("Arial", 10))
        self.lbl_crop.pack(pady=5)
        
        tk.Button(ctrl_frame, text="Reset Crop", command=self.reset_crop, width=15).pack(pady=2)

        tk.Frame(ctrl_frame, height=2, bd=1, relief=tk.SUNKEN).pack(fill=tk.X, pady=10)

        # [3] Task Segmentation (5단계)
        tk.Label(ctrl_frame, text="[3] Task Segmentation (5 Tasks)", font=("Arial", 11, "bold")).pack(pady=10)
        
        # Split 1: End of Needle_Pickup
        self.btn_split1 = tk.Button(ctrl_frame, text="End of Needle_Pickup\n(→ Handoff)", command=self.set_split_1, bg="#FFE4B5")
        self.btn_split1.pack(pady=3, fill=tk.X, padx=20)
        self.lbl_split1 = tk.Label(ctrl_frame, text="Split 1: 0", font=("Arial", 9))
        self.lbl_split1.pack()

        # Split 2: End of Needle_Handoff
        self.btn_split2 = tk.Button(ctrl_frame, text="End of Needle_Handoff\n(→ Insertion)", command=self.set_split_2, bg="#98FB98")
        self.btn_split2.pack(pady=3, fill=tk.X, padx=20)
        self.lbl_split2 = tk.Label(ctrl_frame, text="Split 2: 0", font=("Arial", 9))
        self.lbl_split2.pack()

        # Split 3: End of Needle_Insertion
        self.btn_split3 = tk.Button(ctrl_frame, text="End of Needle_Insertion\n(→ Extraction)", command=self.set_split_3, bg="#87CEEB")
        self.btn_split3.pack(pady=3, fill=tk.X, padx=20)
        self.lbl_split3 = tk.Label(ctrl_frame, text="Split 3: 0", font=("Arial", 9))
        self.lbl_split3.pack()

        # Split 4: End of Needle_Extraction
        self.btn_split4 = tk.Button(ctrl_frame, text="End of Needle_Extraction\n(→ Knot_Tying)", command=self.set_split_4, bg="#DDA0DD")
        self.btn_split4.pack(pady=3, fill=tk.X, padx=20)
        self.lbl_split4 = tk.Label(ctrl_frame, text="Split 4: 0", font=("Arial", 9))
        self.lbl_split4.pack()

        self.lbl_curr_task = tk.Label(ctrl_frame, text="Task: Needle_Pickup", font=("Arial", 12, "bold"), fg="orange")
        self.lbl_curr_task.pack(side=tk.BOTTOM, pady=20)

        # Bottom Timeline
        bottom_frame = tk.Frame(work_frame)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

        self.btn_play = tk.Button(bottom_frame, text="▶ Play", command=self.toggle_play, width=8, font=("Arial", 10, "bold"))
        self.btn_play.pack(side=tk.LEFT, padx=(0, 10))

        self.lbl_frame_idx = tk.Label(bottom_frame, text="0 / 0", width=12)
        self.lbl_frame_idx.pack(side=tk.LEFT)

        slider_area = tk.Frame(bottom_frame)
        slider_area.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        
        self.timeline_canvas = tk.Canvas(slider_area, height=15, bg="#ddd", highlightthickness=0)
        self.timeline_canvas.pack(side=tk.TOP, fill=tk.X, expand=True, pady=(0, 2))
        self.timeline_canvas.bind("<Configure>", lambda e: self.draw_timeline())

        self.slider = tk.Scale(slider_area, from_=0, to=100, orient=tk.HORIZONTAL, command=self.on_slider_move, showvalue=0)
        self.slider.pack(side=tk.TOP, fill=tk.X, expand=True)

        # Key Bindings
        self.root.bind("<Left>", lambda e: self.move_frame(-1))
        self.root.bind("<Right>", lambda e: self.move_frame(1))
        self.root.bind("<space>", lambda e: self.toggle_play())
        self.root.bind("<s>", lambda e: self.save_data())

    # --- File Loading ---
    def load_folder(self):
        path = filedialog.askdirectory()
        if not path: return
        self.data_dir = path
        
        all_files = sorted(glob.glob(os.path.join(path, "*.h5")))
        valid_files = []
        corrupted_files = []

        self.root.config(cursor="watch") 
        self.root.update()

        for f in all_files:
            if self.check_file_integrity(f):
                valid_files.append(f)
            else:
                corrupted_files.append(f)

        self.root.config(cursor="")

        self.file_list = valid_files
        self.file_listbox.delete(0, tk.END)
        for f in valid_files:
            self.file_listbox.insert(tk.END, os.path.basename(f))

        if corrupted_files:
            display_list = corrupted_files[:10]
            msg_detail = "\n".join([f"- {os.path.basename(f)}" for f in display_list])
            
            if len(corrupted_files) > 10:
                msg_detail += f"\n... and {len(corrupted_files)-10} more."
                
            full_msg = f"Found {len(corrupted_files)} corrupted files.\n(These cannot be opened)\n\n{msg_detail}"
            messagebox.showwarning("Corrupted Files Found", full_msg)
        
        if valid_files:
            self.file_listbox.selection_set(0)
            self.on_file_select(None)

    def check_file_integrity(self, file_path):
        try:
            with h5py.File(file_path, 'r') as f:
                _ = f.keys()
            return True
        except:
            return False

    def on_file_select(self, event):
        sel = self.file_listbox.curselection()
        if not sel: return
        idx = sel[0]
        file_path = self.file_list[idx]
        if self.current_file_path == file_path: return
        self.close_current_file()
        self.load_h5(file_path)

    def close_current_file(self):
        self.stop_play()
        if self.current_h5:
            self.current_h5.close()
            self.current_h5 = None
        self.step_keys = []
        self.current_file_path = ""

    def load_h5(self, path):
        try:
            self.current_h5 = h5py.File(path, 'r+') 
            self.current_file_path = path
            self.lbl_status.config(text=os.path.basename(path))

            self.step_keys = sorted([k for k in self.current_h5.keys() if k.isdigit()], key=int)
            self.total_frames = len(self.step_keys)
            
            self.slider.config(to=self.total_frames - 1)
            self.slider.set(0)
            
            self.crop_start = 0
            self.crop_end = max(0, self.total_frames - 1)
            
            self.restore_annotations()
            self.draw_timeline()
            self.update_image(0)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open file: {e}")
            self.close_current_file()

    def restore_annotations(self):
        self.insertion_pt = None
        self.exit_pt = None
        
        # [변경] 5등분 초기화
        self.split_idx_1 = self.total_frames // 5
        self.split_idx_2 = (self.total_frames // 5) * 2
        self.split_idx_3 = (self.total_frames // 5) * 3
        self.split_idx_4 = (self.total_frames // 5) * 4

        if not self.step_keys: return
        
        first_step = self.current_h5[self.step_keys[0]]
        
        if 'needle_location' in first_step:
            try:
                json_str = first_step['needle_location'][()].decode('utf-8')
                data = json.loads(json_str)
                self.insertion_pt = tuple(data.get('insertion')) if data.get('insertion') else None
                self.exit_pt = tuple(data.get('exit')) if data.get('exit') else None
            except: pass
        
        # [변경] 4개의 split 복원
        if 'tool_splits' in self.current_h5.attrs:
            splits = self.current_h5.attrs['tool_splits']
            if len(splits) >= 4:
                self.split_idx_1 = int(splits[0])
                self.split_idx_2 = int(splits[1])
                self.split_idx_3 = int(splits[2])
                self.split_idx_4 = int(splits[3])
            elif len(splits) == 2:
                # 기존 2-split 파일 호환
                self.split_idx_1 = int(splits[0]) // 2
                self.split_idx_2 = int(splits[0])
                self.split_idx_3 = (int(splits[0]) + int(splits[1])) // 2
                self.split_idx_4 = int(splits[1])
        
        self.update_ui_labels()

    # --- Timeline & Crop Visuals ---
    def draw_timeline(self):
        if self.total_frames == 0: return
        w = self.timeline_canvas.winfo_width()
        h = self.timeline_canvas.winfo_height()
        self.timeline_canvas.delete("all")
        
        def get_x(idx):
            return int(w * (idx / (self.total_frames - 1))) if self.total_frames > 1 else 0

        c_start_x = get_x(self.crop_start)
        c_end_x = get_x(self.crop_end)
        s1_x = get_x(self.split_idx_1)
        s2_x = get_x(self.split_idx_2)
        s3_x = get_x(self.split_idx_3)
        s4_x = get_x(self.split_idx_4)
        
        # Background (cropped area)
        self.timeline_canvas.create_rectangle(0, 0, w, h, fill="#888", outline="")
        
        # [변경] 5개 구간 색상: Pickup(orange), Handoff(green), Insertion(blue), Extraction(purple), Knot(red)
        colors = ["#FFB347", "#98FB98", "#87CEEB", "#DDA0DD", "#FF6B6B"]
        splits = [c_start_x, s1_x, s2_x, s3_x, s4_x, c_end_x]
        
        for i in range(5):
            start = max(splits[i], c_start_x)
            end = min(splits[i+1], c_end_x)
            if end > start:
                self.timeline_canvas.create_rectangle(start, 0, end, h, fill=colors[i], outline="")

    # --- Crop Logic ---
    def set_crop_start(self):
        if self.current_frame_idx >= self.crop_end: return
        self.crop_start = self.current_frame_idx
        self.update_ui_labels()
        self.draw_timeline()

    def set_crop_end(self):
        if self.current_frame_idx <= self.crop_start: return
        self.crop_end = self.current_frame_idx
        self.update_ui_labels()
        self.draw_timeline()

    def reset_crop(self):
        self.crop_start = 0
        self.crop_end = self.total_frames - 1
        self.update_ui_labels()
        self.draw_timeline()

    # --- [변경] Tasks Logic: 4개의 split 설정 ---
    def set_split_1(self):
        self.split_idx_1 = self.current_frame_idx
        if self.split_idx_2 <= self.split_idx_1: self.split_idx_2 = self.split_idx_1 + 1
        if self.split_idx_3 <= self.split_idx_2: self.split_idx_3 = self.split_idx_2 + 1
        if self.split_idx_4 <= self.split_idx_3: self.split_idx_4 = self.split_idx_3 + 1
        self.update_ui_labels()
        self.draw_timeline()
        self.update_image(self.current_frame_idx)

    def set_split_2(self):
        if self.current_frame_idx <= self.split_idx_1: return
        self.split_idx_2 = self.current_frame_idx
        if self.split_idx_3 <= self.split_idx_2: self.split_idx_3 = self.split_idx_2 + 1
        if self.split_idx_4 <= self.split_idx_3: self.split_idx_4 = self.split_idx_3 + 1
        self.update_ui_labels()
        self.draw_timeline()
        self.update_image(self.current_frame_idx)

    def set_split_3(self):
        if self.current_frame_idx <= self.split_idx_2: return
        self.split_idx_3 = self.current_frame_idx
        if self.split_idx_4 <= self.split_idx_3: self.split_idx_4 = self.split_idx_3 + 1
        self.update_ui_labels()
        self.draw_timeline()
        self.update_image(self.current_frame_idx)

    def set_split_4(self):
        if self.current_frame_idx <= self.split_idx_3: return
        self.split_idx_4 = self.current_frame_idx
        self.update_ui_labels()
        self.draw_timeline()
        self.update_image(self.current_frame_idx)

    # --- UI Updates ---
    def update_ui_labels(self):
        self.lbl_ins.config(text=f"Insertion: {self.insertion_pt}" if self.insertion_pt else "Insertion: None")
        self.lbl_exit.config(text=f"Exit: {self.exit_pt}" if self.exit_pt else "Exit: None")
        self.lbl_split1.config(text=f"End Pickup: {self.split_idx_1}")
        self.lbl_split2.config(text=f"End Handoff: {self.split_idx_2}")
        self.lbl_split3.config(text=f"End Insertion: {self.split_idx_3}")
        self.lbl_split4.config(text=f"End Extraction: {self.split_idx_4}")
        self.lbl_crop.config(text=f"Valid: {self.crop_start} ~ {self.crop_end}")

    def update_image(self, idx):
        if not self.step_keys or idx >= len(self.step_keys): return
        self.current_frame_idx = idx
        
        try:
            key = self.step_keys[idx]
            img_data = self.current_h5[key]['image1'][()]
            
            if img_data.size == 0:
                img_np = np.zeros((240, 320, 3), dtype=np.uint8)
            else:
                arr = np.frombuffer(img_data, np.uint8)
                img_np = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            
            pil_img = Image.fromarray(img_np)
            orig_w, orig_h = pil_img.size
            new_w, new_h = int(orig_w * self.view_scale), int(orig_h * self.view_scale)
            
            resized_pil = pil_img.resize((new_w, new_h), Image.Resampling.BILINEAR)
            self.tk_img = ImageTk.PhotoImage(resized_pil)
            
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)
            self.canvas.config(scrollregion=(0, 0, new_w, new_h))
            
            # Draw Points
            r = 6
            if self.insertion_pt:
                rx, ry = self.insertion_pt
                vx, vy = rx * self.view_scale, ry * self.view_scale
                self.canvas.create_oval(vx-r, vy-r, vx+r, vy+r, fill='lime', outline='black', width=2)
                self.canvas.create_text(vx, vy-20, text="Insertion", fill="lime", font=("Bold", 12))
                
            if self.exit_pt:
                rx, ry = self.exit_pt
                vx, vy = rx * self.view_scale, ry * self.view_scale
                self.canvas.create_oval(vx-r, vy-r, vx+r, vy+r, fill='red', outline='black', width=2)
                self.canvas.create_text(vx, vy-20, text="Exit", fill="red", font=("Bold", 12))

            # [변경] Task Text (5단계)
            if idx < self.crop_start or idx > self.crop_end:
                task, color = "EXCLUDED (CROP)", "gray"
            else:
                if idx < self.split_idx_1: task, color = "Needle_Pickup", "#FF8C00"
                elif idx < self.split_idx_2: task, color = "Needle_Handoff", "#228B22"
                elif idx < self.split_idx_3: task, color = "Needle_Insertion", "#4169E1"
                elif idx < self.split_idx_4: task, color = "Needle_Extraction", "#9932CC"
                else: task, color = "Knot_Tying", "#DC143C"

            self.lbl_curr_task.config(text=f"Task: {task}", fg=color)
            self.lbl_frame_idx.config(text=f"{idx} / {self.total_frames-1}")
            
        except Exception as e:
            print(f"Image load error: {e}")

    # --- Interaction ---
    def set_mode(self, mode): self.click_mode = mode
    def on_canvas_click(self, event):
        x_raw, y_raw = event.x, event.y
        real_x = int(x_raw / self.view_scale)
        real_y = int(y_raw / self.view_scale)
        if self.click_mode == 'insertion':
            self.insertion_pt = (real_x, real_y); self.set_mode('none')
        elif self.click_mode == 'exit':
            self.exit_pt = (real_x, real_y); self.set_mode('none')
        self.update_ui_labels(); self.update_image(self.current_frame_idx)
    
    def on_slider_move(self, val): self.update_image(int(val))
    def move_frame(self, delta):
        new_idx = max(0, min(self.current_frame_idx + delta, self.total_frames - 1))
        self.slider.set(new_idx)
    def toggle_play(self):
        if self.is_playing: self.stop_play()
        else: self.start_play()
    def start_play(self):
        self.is_playing = True; self.btn_play.config(text="⏸ Pause"); self.play_step()
    def stop_play(self):
        self.is_playing = False; self.btn_play.config(text="▶ Play")
    def play_step(self):
        if not self.is_playing: return
        next_frame = self.current_frame_idx + 1
        if next_frame >= self.total_frames: self.stop_play(); return
        self.slider.set(next_frame)
        self.root.after(self.play_speed_ms, self.play_step)

    # --- SMART SAVE LOGIC ---
    def save_data(self):
        if not self.current_h5: return
        self.stop_play()
        
        if self.insertion_pt is None or self.exit_pt is None:
            if not messagebox.askyesno("Warning", "Needle points not set. Save anyway?"): return

        try:
            self.root.config(cursor="watch")
            self.root.update()

            loc_dict = {"insertion": self.insertion_pt, "exit": self.exit_pt}
            loc_json = json.dumps(loc_dict)
            str_dt = h5py.string_dtype(encoding='utf-8')
            
            is_cropped = not (self.crop_start == 0 and self.crop_end == self.total_frames - 1)
            
            if is_cropped:
                # [CASE 1] Crop: 전체 파일 재작성
                dir_name = os.path.dirname(self.current_file_path)
                temp_path = os.path.join(dir_name, "temp_save.h5")
                
                with h5py.File(temp_path, 'w') as f_out:
                    new_idx = 0
                    for old_idx in range(self.crop_start, self.crop_end + 1):
                        old_key = self.step_keys[old_idx]
                        old_grp = self.current_h5[old_key]
                        new_grp = f_out.create_group(str(new_idx))
                        
                        for k in old_grp.keys():
                            if k in ['task', 'needle_location']: continue
                            old_grp.copy(k, new_grp)
                        
                        task_str = self.get_task_str_for_idx(old_idx)
                        new_grp.create_dataset('task', data=task_str, dtype=str_dt)
                        new_grp.create_dataset('needle_location', data=loc_json, dtype=str_dt)
                        new_idx += 1
                    
                    # [변경] 4개 split 저장
                    new_split1 = max(0, self.split_idx_1 - self.crop_start)
                    new_split2 = max(0, self.split_idx_2 - self.crop_start)
                    new_split3 = max(0, self.split_idx_3 - self.crop_start)
                    new_split4 = max(0, self.split_idx_4 - self.crop_start)
                    f_out.attrs['tool_splits'] = np.array([new_split1, new_split2, new_split3, new_split4])
                
                self.current_h5.close()
                self.current_h5 = None
                shutil.move(temp_path, self.current_file_path)
                print(f"File cropped & replaced: {self.current_file_path}")

            else:
                # [CASE 2] No Crop: In-place Update
                print("Smart Save: Updating metadata only...")
                
                for idx in range(self.total_frames):
                    key = self.step_keys[idx]
                    grp = self.current_h5[key]
                    
                    task_str = self.get_task_str_for_idx(idx)
                    
                    if 'task' in grp: del grp['task']
                    if 'needle_location' in grp: del grp['needle_location']
                    
                    grp.create_dataset('task', data=task_str, dtype=str_dt)
                    grp.create_dataset('needle_location', data=loc_json, dtype=str_dt)
                
                # [변경] 4개 split 저장
                self.current_h5.attrs['tool_splits'] = np.array([self.split_idx_1, self.split_idx_2, self.split_idx_3, self.split_idx_4])
                self.current_h5.flush()

            self.load_h5(self.current_file_path)
            self.root.config(cursor="")
            status_msg = "SAVED (Cropped)" if is_cropped else "SAVED (Instant)"
            self.lbl_status.config(text=f"{status_msg}: {os.path.basename(self.current_file_path)}")
            
        except Exception as e:
            self.root.config(cursor="")
            messagebox.showerror("Save Error", str(e))
            if not self.current_h5:
                try: self.load_h5(self.current_file_path)
                except: pass

    # [변경] 5단계 태스크 문자열 반환
    def get_task_str_for_idx(self, idx):
        if idx < self.split_idx_1: return "Needle_Pickup"
        elif idx < self.split_idx_2: return "Needle_Handoff"
        elif idx < self.split_idx_3: return "Needle_Insertion"
        elif idx < self.split_idx_4: return "Needle_Extraction"
        else: return "Knot_Tying"

if __name__ == "__main__":
    root = tk.Tk()
    app = SurgicalAnnotationTool(root)
    root.mainloop()
    