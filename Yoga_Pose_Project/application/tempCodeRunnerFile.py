import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import ttk, font
from PIL import Image, ImageTk
import numpy as np
import threading
import requests
from io import BytesIO
import pandas as pd
import json
import os

class YogaPoseApp:
    def __init__(self, root):
        """
        Initializes the main application window, variables, and UI components.
        """
        self.root = root
        self.root.title("YPDS - Yoga Pose Detection System")
        self.root.geometry("1300x800")
        self.root.configure(bg="#0f172a")

        # --- State Variables ---
        self.is_camera_on = False
        self.selected_pose = None
        self.cap = None
        self.video_thread = None
        self.stop_thread = False

        # --- MediaPipe Initialization ---
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # --- Load Pose Data from CSV ---
        self.reference_poses = self.load_poses_from_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yoga_poses.csv'))
        self.reference_images = {}
        if self.reference_poses:
            self.load_reference_images()

        # --- UI Setup ---
        self.setup_styles()
        self.setup_ui()

    def load_poses_from_csv(self, csv_path):
        if not os.path.exists(csv_path):
            print(f"Error: Pose data file not found at '{csv_path}'!")
            tk.messagebox.showerror("File Not Found", f"Could not find the pose data file:\n{csv_path}")
            return {}
        try:
            df = pd.read_csv(csv_path, dtype={'landmarks': str})
        except Exception as e:
            print(f"Error reading CSV: {e}")
            tk.messagebox.showerror("CSV Error", f"Could not read the pose data file:\n{csv_path}\n\nError: {e}")
            return {}
        poses_dict = {}
        
        for _, row in df.iterrows():
        
            pose_name = row.get('name', 'Unnamed Pose')
            try:
                landmarks_str = row['landmarks']
                if pd.isna(landmarks_str) or not landmarks_str.strip():
                    print(f"Warning: Skipping pose '{pose_name}' due to missing landmark data.")
                    continue

                landmarks = json.loads(landmarks_str)
                
                for i, lm in enumerate(landmarks):
                    if 'name' not in lm:
                        lm['name'] = f"Joint Angle {i+1}"
                
                poses_dict[pose_name] = {
                    'image_url': row['image_url'],
                    'landmarks': landmarks
                }
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                print(f"Warning: Could not parse landmarks for pose '{pose_name}'. Error: {e}")
        
        if not poses_dict:
            print("Error: No valid poses were loaded. Please check the CSV file format.")
            
        return poses_dict

    def setup_styles(self):
        """Configures the styles for ttk widgets."""
        style = ttk.Style()
        style.theme_use('clam')
        
        BG_COLOR = "#0f172a"
        TEXT_COLOR = "#e2e8f0"
        PRIMARY_COLOR = "#38bdf8"
        SECONDARY_COLOR = "#475569"
        SUCCESS_COLOR = "#22c55e"
        WARNING_COLOR = "#facc15"
        ERROR_COLOR = "#ef4444"
        
        self.header_font = font.Font(family="Inter", size=24, weight="bold")
        self.subheader_font = font.Font(family="Inter", size=12)
        self.bold_font = font.Font(family="Inter", size=10, weight="bold")
        self.normal_font = font.Font(family="Inter", size=10)

        style.configure("TFrame", background=BG_COLOR)
        style.configure("TLabel", background=BG_COLOR, foreground=TEXT_COLOR, font=self.normal_font)
        style.configure("Header.TLabel", font=self.header_font, foreground=PRIMARY_COLOR)
        style.configure("SubHeader.TLabel", font=self.subheader_font, foreground=SECONDARY_COLOR)
        
        style.configure("TNotebook", background=BG_COLOR, borderwidth=0)
        style.configure("TNotebook.Tab", background=SECONDARY_COLOR, foreground=TEXT_COLOR, font=self.bold_font, padding=[10, 5])
        style.map("TNotebook.Tab", background=[("selected", PRIMARY_COLOR), ("active", SECONDARY_COLOR)])

        style.configure("Pose.TButton", font=self.bold_font, background="#334155", foreground=TEXT_COLOR, borderwidth=0, padding=10)
        style.map("Pose.TButton", background=[('active', SECONDARY_COLOR), ('selected', PRIMARY_COLOR)])
        
        style.configure("Start.TButton", font=self.bold_font, background=PRIMARY_COLOR, foreground=TEXT_COLOR, padding=10)
        style.map("Start.TButton", background=[('active', "#78f967")])
        
        style.configure("Stop.TButton", font=self.bold_font, background=ERROR_COLOR, foreground=TEXT_COLOR, padding=10)
        style.map("Stop.TButton", background=[('active', '#f87171')])
        
        style.configure("green.Horizontal.TProgressbar", background=SUCCESS_COLOR)
        style.configure("yellow.Horizontal.TProgressbar", background=WARNING_COLOR)
        style.configure("red.Horizontal.TProgressbar", background=ERROR_COLOR)
        
    def setup_ui(self):
        """Creates and arranges the UI components."""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        left_panel = ttk.Frame(main_frame, width=900)
        right_panel = ttk.Frame(main_frame, padding=(20, 10))
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        self.video_label = ttk.Label(left_panel, background="black", anchor=tk.CENTER)
        self.video_label.pack(fill=tk.BOTH, expand=True)

        header = ttk.Label(right_panel, text="Yoga Pose Detection System", style="Header.TLabel")
        header.pack(pady=(0, 5), anchor="w")
        subheader = ttk.Label(right_panel, text="Your Personal Wellness Companion", style="SubHeader.TLabel")
        subheader.pack(pady=(0, 20), anchor="w")
        
        notebook = ttk.Notebook(right_panel)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        pose_tab = ttk.Frame(notebook, padding=10)
        efficiency_tab = ttk.Frame(notebook, padding=10)
        
        notebook.add(pose_tab, text="🧘 Pose Controls")
        notebook.add(efficiency_tab, text="🚀 Efficiency Report")
        
        self.setup_pose_tab(pose_tab)
        self.setup_efficiency_tab(efficiency_tab)
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        if self.reference_poses:
            self.select_pose(list(self.reference_poses.keys())[0])

    def setup_pose_tab(self, parent_tab):
        """Sets up the content for the pose selection and feedback tab."""
        pose_frame = ttk.Frame(parent_tab)
        pose_frame.pack(fill=tk.X, pady=10)
        ttk.Label(pose_frame, text="Choose a Pose", font=("Inter", 14, "bold")).pack(anchor="w", pady=(0, 10))
        
        self.pose_buttons = {}
        btn_container = ttk.Frame(pose_frame)
        btn_container.pack(fill=tk.X)
        if not self.reference_poses:
            # Show message if no poses loaded
            ttk.Label(btn_container, text="No poses loaded. Please check yoga_poses.csv.", foreground="#ef4444", font=("Inter", 12, "bold")).pack(anchor="center", pady=10)
        else:
            for i, pose_name in enumerate(self.reference_poses.keys()):
                btn = ttk.Button(btn_container, text=pose_name, style="Pose.TButton", command=lambda p=pose_name: self.select_pose(p))
                btn.grid(row=i//2, column=i%2, sticky="ew", padx=2, pady=2)
                self.pose_buttons[pose_name] = btn
            btn_container.grid_columnconfigure((0,1), weight=1)

        feedback_panel = ttk.Frame(parent_tab, padding=10)
        feedback_panel.pack(fill=tk.BOTH, expand=True, pady=20)
        
        ttk.Label(feedback_panel, text="Live Feedback", font=("Inter", 14, "bold")).pack(anchor="w")
        
        ref_image_frame = ttk.Frame(feedback_panel)
        ref_image_frame.pack(pady=10, fill=tk.X)
        self.ref_image_label = ttk.Label(ref_image_frame, background="#0f172a")
        self.ref_image_label.pack(side=tk.LEFT, padx=(0, 10))
        
        pose_info_frame = ttk.Frame(ref_image_frame)
        pose_info_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.current_pose_label = ttk.Label(pose_info_frame, text="No Pose Selected", font=("Inter", 12, "bold"), foreground="#38bdf8")
        self.current_pose_label.pack(anchor="w")
        
        self.feedback_text_label = ttk.Label(pose_info_frame, text="Start the camera to begin.", wraplength=200, justify=tk.LEFT)
        self.feedback_text_label.pack(anchor="w", fill=tk.X)

        ttk.Label(feedback_panel, text="Posture Accuracy").pack(anchor="w", pady=(20, 5))
        self.progress_bar = ttk.Progressbar(feedback_panel, orient="horizontal", length=300, mode="determinate")
        self.progress_bar.pack(fill=tk.X, pady=2)
        self.score_label = ttk.Label(feedback_panel, text="0%", font=("Courier", 10))
        self.score_label.pack(anchor="e")

        self.start_stop_btn = ttk.Button(parent_tab, text="Start Camera", style="Start.TButton", command=self.toggle_camera)
        self.start_stop_btn.pack(fill=tk.X, pady=10, side=tk.BOTTOM)

    def setup_efficiency_tab(self, parent_tab):
        """Sets up the content for the efficiency report tab."""
        ttk.Label(parent_tab, text="Body Detection Performance", font=("Inter", 14, "bold")).pack(anchor="w", pady=(0, 15))
        
        report_text = (
            "This app uses Google's MediaPipe Pose model for real-time body tracking.\n\n"
            "Key Efficiency Metrics:\n"
            "• Speed: Runs at 30+ FPS on most modern devices, ensuring smooth, real-time feedback.\n\n"
            "• Accuracy: Achieves a high Probability of Correct Keypoint (PCK) score, making it highly reliable in locating body joints.\n\n"
            "• Architecture: Employs a two-step pipeline (BlazePose Detector + Tracker) for an optimal balance between speed and accuracy.\n\n"
            "• CPU/GPU Usage: Highly optimized for low resource consumption, suitable for a wide range of devices.\n\n"
            "Limitations:\n"
            "Accuracy may decrease if body parts are hidden (occluded) or in unusual lighting conditions."
        )
        
        report_label = ttk.Label(parent_tab, text=report_text, wraplength=350, justify=tk.LEFT)
        report_label.pack(anchor="w", fill=tk.X)
    
    def load_reference_images(self):
        def loader():
            for name, data in self.reference_poses.items():
                try:
                    response = requests.get(data['image_url'])
                    img_data = response.content
                    img = Image.open(BytesIO(img_data)).resize((200, 200), Image.LANCZOS)
                    self.reference_images[name] = ImageTk.PhotoImage(img)
                except Exception as e:
                    print(f"Error loading image for {name}: {e}")
                    self.reference_images[name] = None
            self.root.after(0, self.update_ref_image_ui)

        threading.Thread(target=loader, daemon=True).start()

    def update_ref_image_ui(self):
        if self.selected_pose and self.reference_images.get(self.selected_pose):
             self.ref_image_label.config(image=self.reference_images[self.selected_pose])

    def select_pose(self, pose_name):
        self.selected_pose = pose_name
        self.current_pose_label.config(text=pose_name)
        self.reset_feedback()
        
        if self.reference_images.get(pose_name):
            self.ref_image_label.config(image=self.reference_images[pose_name])

        for name, btn in self.pose_buttons.items():
            btn.state(['!selected'] if name != pose_name else ['selected'])

    def toggle_camera(self):
        if not self.is_camera_on:
            try:
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    raise IOError("Cannot open webcam")
                self.is_camera_on = True
                self.stop_thread = False
                self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
                self.video_thread.start()
                self.start_stop_btn.config(text="Stop Camera", style="Stop.TButton")
            except IOError as e:
                self.feedback_text_label.config(text=f"Error: {e}")
        else:
            self.is_camera_on = False
            self.start_stop_btn.config(text="Start Camera", style="Start.TButton")
            self.stop_thread = True
            if self.video_thread: self.video_thread.join(timeout=1)
            if self.cap: self.cap.release()
            self.video_label.config(image=''); self.video_label.image = None

    def video_loop(self):
        while self.cap.isOpened() and not self.stop_thread:
            ret, frame = self.cap.read()
            if not ret: break
            
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)
            
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())

                if self.selected_pose:
                    score, feedback_msg = self.compare_poses(results.pose_landmarks.landmark, self.reference_poses[self.selected_pose])
                    self.root.after(0, self.update_feedback, score, feedback_msg)
            
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img_tk = ImageTk.PhotoImage(image=img)
            self.root.after(0, self.update_video_label, img_tk)

    def update_video_label(self, img_tk):
        self.video_label.imgtk = img_tk
        self.video_label.configure(image=img_tk)

    def calculate_angle(self, a, b, c):
        a = np.array([a.x, a.y]); b = np.array([b.x, b.y]); c = np.array([c.x, c.y])
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return 360 - angle if angle > 180.0 else angle

    def compare_poses(self, user_landmarks, reference_pose):
        """
        Compares poses using a weighted scoring system based on landmark importance.
        """
        total_score, total_weight = 0, 0
        worst_angle_diff, worst_angle_name = -1, ""
        
        for ref_angle in reference_pose['landmarks']:
            lm_a = user_landmarks[ref_angle['a']]
            lm_b = user_landmarks[ref_angle['b']]
            lm_c = user_landmarks[ref_angle['c']]
            weight = ref_angle.get('weight', 1.0) # Default weight is 1 if not specified

            # Check if all three landmarks for the angle are clearly visible
            if lm_a.visibility > 0.7 and lm_b.visibility > 0.7 and lm_c.visibility > 0.7:
                total_weight += weight
                user_angle = self.calculate_angle(lm_a, lm_b, lm_c)
                angle_diff = abs(user_angle - ref_angle['angle'])
                
                # Use a weighted difference to find the most critical error
                if angle_diff * weight > worst_angle_diff:
                    worst_angle_diff = angle_diff * weight
                    worst_angle_name = ref_angle.get('name', 'a key joint')

                # Calculate score for this angle and apply its weight
                angle_score = max(0, 1 - (angle_diff / 50)) # A bit more lenient on difference
                total_score += angle_score * weight
        
        # If no joints are visible, return a 0 score
        if total_weight == 0:
            return 0, "Move into the frame to get feedback."

        # Calculate the final score as a weighted average
        final_score = (total_score / total_weight) * 100
        
        # Generate feedback message
        feedback_msg = "Adjust your pose to match the guide."
        if final_score > 95:
            feedback_msg = "Perfect Form! Hold the pose."
        elif final_score > 80:
            feedback_msg = "Excellent! A few minor adjustments."
        elif final_score > 60:
            feedback_msg = f"Good! Focus on adjusting your {worst_angle_name}."
        
        return final_score, feedback_msg

    def update_feedback(self, score, feedback_msg):
        score_int = int(score)
        self.progress_bar['value'] = score_int
        self.score_label.config(text=f"{score_int}%")
        self.feedback_text_label.config(text=feedback_msg)
        
        if score_int > 50: self.progress_bar.config(style="green.Horizontal.TProgressbar")
        elif score_int > 30: self.progress_bar.config(style="yellow.Horizontal.TProgressbar")
        else: self.progress_bar.config(style="red.Horizontal.TProgressbar")

    def reset_feedback(self):
        self.progress_bar['value'] = 0
        self.progress_bar.config(style="TProgressbar")
        self.score_label.config(text="0%")
        self.feedback_text_label.config(text="Start the camera to begin." if not self.is_camera_on else "Get into the pose.")

    def on_closing(self):
        """Handles the application closing event to safely stop threads and release resources."""
        self.stop_thread = True
        if self.video_thread:
            self.video_thread.join(timeout=1)
        if self.cap:
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = YogaPoseApp(root)
    root.mainloop()