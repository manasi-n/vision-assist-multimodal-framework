# vision_assist_app.py
# Vision Assist — Object Detection + Speech Alerts + Image Captioning + Translation
# Includes robust camera loop with per-object tracking (IoU), repeated alerts, and fallbacks.

import os
import cv2
import numpy as np
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from googletrans import Translator
from PIL import Image, ImageTk
import pyttsx3
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from threading import Thread, Event
from ultralytics import YOLO

# =========================
# Initialize models (graceful failures)
# =========================
# BLIP
try:
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
except Exception as e:
    processor = None
    blip_model = None
    print("BLIP load error (captioning disabled):", e)

# Translator
try:
    translator = Translator()
except Exception as e:
    translator = None
    print("Translator load error:", e)

# TTS
try:
    tts_engine = pyttsx3.init()
except Exception as e:
    tts_engine = None
    print("pyttsx3 init error (TTS disabled):", e)

# YOLO
try:
    yolo_model = YOLO("yolov8n.pt")
except Exception:
    try:
        yolo_model = YOLO("yolov8n")
    except Exception as e:
        yolo_model = None
        print("YOLO load error (detection disabled):", e)

# MiDaS (optional)
midas = None
transform = None
try:
    if torch is not None:
        midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        midas.to("cuda" if torch.cuda.is_available() else "cpu")
        midas.eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        transform = midas_transforms.small_transform
except Exception as e:
    midas = None
    transform = None
    print("MiDaS load error (depth disabled):", e)

calibration = {"scale": 10.0}  # used for depth->meters conversion (adjust via calibrate)

# =========================
# Helper functions
# =========================
def speak_text(text):
    """Non-blocking TTS (pyttsx3) with console fallback."""
    print("🔊 TTS REQUEST:", text)
    if tts_engine is None:
        print("TTS not available; would speak:", text)
        return
    def _s():
        try:
            tts_engine.say(text)
            tts_engine.runAndWait()
        except Exception as e:
            print("TTS error:", e)
    Thread(target=_s, daemon=True).start()

def process_image_for_caption(image_path):
    if processor is None or blip_model is None:
        return "Captioning model not available."
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        output = blip_model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

def translate_text(text):
    translations = {"English": text}
    if translator is None:
        translations["Telugu"] = "Translator not available"
        translations["Hindi"] = "Translator not available"
        return translations
    try:
        translations["Telugu"] = translator.translate(text, src='en', dest='te').text
        translations["Hindi"] = translator.translate(text, src='en', dest='hi').text
    except Exception:
        translations["Telugu"] = "Translation Error"
        translations["Hindi"] = "Translation Error"
    return translations

def estimate_depth_map(frame_rgb):
    """Return normalized depth map 0..1 using MiDaS (raises if unavailable)."""
    if midas is None or transform is None:
        raise RuntimeError("MiDaS not available")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_batch = transform(frame_rgb).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame_rgb.shape[:2],
            mode='bicubic',
            align_corners=False
        ).squeeze()
        depth = prediction.cpu().numpy()
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    return depth

def estimate_distance_from_depth(depth_crop):
    median_depth = np.median(depth_crop)
    if median_depth <= 0 or not np.isfinite(median_depth):
        return None
    scale = calibration.get("scale", 10.0)
    return float(scale / (median_depth + 1e-6))

def estimate_distance_from_bbox_proxy(frame_shape, bbox, proxy_scale=1.0):
    """Coarse fallback: inverse proportional to bounding-box height fraction."""
    h_frame = frame_shape[0]
    x1, y1, x2, y2 = bbox
    bbox_h = max(1, y2 - y1)
    rel = bbox_h / float(h_frame)
    if rel <= 0:
        return None
    scale = calibration.get("scale", 10.0) * proxy_scale
    return float(scale / (rel + 1e-6))

# =========================
# GUI setup
# =========================
root = tk.Tk()
root.title("✨ Vision Assist — AI Blind Assistance")
root.geometry("1200x800")
root.configure(bg="#eef7f7")

header = tk.Label(root, text="🧠 Image Captioning + Translation + Live Distance Alert",
                  font=("Helvetica", 18, "bold"), bg="#0aa1a1", fg="#ffffff", pady=8)
header.pack(fill="x")

content = tk.Frame(root, bg="#ffffff", padx=12, pady=12)
content.pack(fill="both", expand=True, padx=12, pady=12)

left = tk.Frame(content, bg="#ffffff")
left.pack(side="left", fill="y", padx=6)

right = tk.Frame(content, bg="#ffffff")
right.pack(side="right", fill="both", expand=True, padx=6)

# Left: image caption controls
select_btn = tk.Button(left, text="📁 Select Image for Captioning", font=("Arial", 12))
select_btn.pack(pady=8)

caption_var = tk.StringVar()
caption_label = tk.Label(left, textvariable=caption_var, wraplength=400, justify="left",
                         bg="#f7f7f7", relief="sunken", bd=1, padx=6, pady=6, font=("Georgia", 12))
caption_label.pack(pady=8)

# Right: video + controls
video_panel = tk.Label(right, bg="#222")
video_panel.pack(padx=6, pady=6)

camera_controls = tk.Frame(right, bg="#ffffff")
camera_controls.pack(fill="x", pady=6)

start_cam_btn = tk.Button(camera_controls, text="▶ Start Camera", font=("Arial", 12))
stop_cam_btn = tk.Button(camera_controls, text="⏹ Stop Camera", font=("Arial", 12), state="disabled")
calib_btn = tk.Button(camera_controls, text="⚙ Calibrate Scale", font=("Arial", 12))
set_thresh_btn = tk.Button(camera_controls, text="Set Alert Threshold (m)", font=("Arial", 12))

start_cam_btn.pack(side="left", padx=6)
stop_cam_btn.pack(side="left", padx=6)
calib_btn.pack(side="left", padx=6)
set_thresh_btn.pack(side="left", padx=6)

alert_threshold = tk.DoubleVar(value=1.5)
threshold_label = tk.Label(right, text=f"Current alert threshold: {alert_threshold.get():.2f} m",
                           bg="#ffffff", font=("Arial", 12))
threshold_label.pack()

# =========================
# Image caption handler
# =========================
def open_and_process_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", ".jpg;.jpeg;.png;.bmp;*.gif")])
    if not file_path:
        return
    def _proc():
        try:
            caption_var.set("⏳ Processing image...")
            caption = process_image_for_caption(file_path)
            translations = translate_text(caption)
            text = f"🌐 English: {translations['English']}\n\n🇮🇳 Telugu: {translations['Telugu']}\n\n🇮🇳 Hindi: {translations['Hindi']}"
            caption_var.set(text)
            speak_text(f"Caption: {translations['English']}")
        except Exception as e:
            messagebox.showerror("Error", str(e))
    Thread(target=_proc, daemon=True).start()

select_btn.config(command=open_and_process_image)

# =========================
# Camera loop (tracking + announcements)
# =========================
cam_thread = None
cam_stop_event = Event()

def camera_loop():
    # Try camera indices 0..3
    print("📸 Attempting to open camera (indices 0..3)...")
    cap = None
    selected_idx = None
    for idx in range(4):
        try:
            test_cap = cv2.VideoCapture(idx)
            ok, frame_test = test_cap.read()
            test_cap.release()
            print(f"  tried index {idx} -> opened={ok}")
            if ok:
                selected_idx = idx
                cap = cv2.VideoCapture(idx)
                print(f"✅ Using camera index {idx}")
                break
        except Exception as e:
            print(f"  index {idx} error:", e)
    if cap is None or not cap.isOpened():
        print("❌ No camera could be opened.")
        try:
            video_panel.configure(image='')
            video_panel_text = tk.Label(right, text="No camera found", bg="#ffffff", fg="red", font=("Arial",12))
            video_panel_text.place(x=300, y=120)
        except Exception:
            pass
        messagebox.showerror("Camera Error", "Could not open the camera. Close other apps using the camera and try again.")
        stop_camera()
        return

    print("✅ Camera opened successfully in camera_loop()")

    # Tracking / announcement parameters
    speak_interval = 2.0       # seconds between repeats for same tracked object
    iou_new_threshold = 0.35   # IoU threshold below which detection is considered a new object
    stale_age = 4.0            # seconds to forget a tracked object if not seen

    tracked = []  # list of dicts: {name, bbox, last_seen, last_announced}

    def iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH
        areaA = max(0, boxA[2]-boxA[0]) * max(0, boxA[3]-boxA[1])
        areaB = max(0, boxB[2]-boxB[0]) * max(0, boxB[3]-boxB[1])
        union = areaA + areaB - interArea
        return interArea / union if union > 0 else 0.0

    while not cam_stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("⚠ Frame read failed; retrying...")
            cv2.waitKey(50)
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # YOLO detection
        detections = []
        if yolo_model is not None:
            try:
                results = yolo_model.predict(frame_rgb, imgsz=640, conf=0.25, verbose=False)
            except Exception as e:
                print("YOLO predict error:", e)
                results = []
            if results and len(results):
                r = results[0]
                boxes = getattr(r, "boxes", [])
                for box in boxes:
                    try:
                        xyxy = box.xyxy.cpu().numpy().flatten()
                        conf = float(box.conf.cpu().numpy())
                        cls = int(box.cls.cpu().numpy())
                        name = yolo_model.names.get(cls, str(cls))
                        detections.append((xyxy, conf, name))
                    except Exception as e:
                        print("Box parse error:", e)
                        continue

        # Depth map (if available)
        try:
            depth_map = estimate_depth_map(frame_rgb) if midas is not None and transform is not None else None
        except Exception as e:
            print("Depth estimate error:", e)
            depth_map = None

        now_time = cv2.getTickCount() / cv2.getTickFrequency()

        # Forget stale tracked objects
        tracked = [t for t in tracked if now_time - t["last_seen"] <= stale_age]

        print(f"Frame: detections_count={len(detections)} depth_map_ok={depth_map is not None} tracked_count={len(tracked)}")

        for (xyxy, conf, name) in detections:
            x1, y1, x2, y2 = map(int, xyxy)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1]-1, x2), min(frame.shape[0]-1, y2)

            # distance estimate
            dist = None
            if depth_map is not None and x2 > x1 and y2 > y1:
                try:
                    crop = depth_map[y1:y2, x1:x2]
                    if crop.size:
                        dist = estimate_distance_from_depth(crop)
                except Exception as e:
                    print("Depth crop error:", e)
                    dist = None
            if dist is None:
                try:
                    dist = estimate_distance_from_bbox_proxy(frame.shape, (x1, y1, x2, y2))
                except Exception:
                    dist = None

            if dist is not None and (not np.isfinite(dist) or dist <= 0):
                dist = None

            print(f"DETECT: {name} conf={conf:.2f} bbox=({x1},{y1},{x2},{y2}) dist={dist}")

            # draw and label
            label = f"{name} {conf:.2f}"
            if dist is not None:
                label += f" {dist:.2f} m"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1, max(15, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            # decide announcement
            should_announce = False
            if dist is None:
                # can't determine distance -> skip announcement
                pass
            else:
                # find best tracked of same class
                best = None
                best_iou = 0.0
                for t in tracked:
                    if t["name"] != name:
                        continue
                    i = iou((x1, y1, x2, y2), t["bbox"])
                    if i > best_iou:
                        best_iou = i
                        best = t

                if best is None or best_iou < iou_new_threshold:
                    # new object
                    should_announce = True
                    tracked.append({
                        "name": name,
                        "bbox": (x1, y1, x2, y2),
                        "last_seen": now_time,
                        "last_announced": 0.0
                    })
                else:
                    # existing tracked object: update last_seen/bbox, maybe re-announce
                    best["last_seen"] = now_time
                    best["bbox"] = (x1, y1, x2, y2)
                    if now_time - best.get("last_announced", 0.0) > speak_interval and dist <= alert_threshold.get():
                        should_announce = True

            if should_announce and dist is not None and dist <= alert_threshold.get():
                msg = f"Alert! {name} at approximately {dist:.1f} meters ahead."
                print("SPEAKING:", msg)
                speak_text(msg)
                # update last_announced of best matching tracked item
                best_idx = None
                best_iou2 = -1.0
                for idx, t in enumerate(tracked):
                    if t["name"] != name:
                        continue
                    i2 = iou((x1, y1, x2, y2), t["bbox"])
                    if i2 > best_iou2:
                        best_iou2 = i2
                        best_idx = idx
                if best_idx is not None:
                    tracked[best_idx]["last_announced"] = now_time

        # display in GUI
        try:
            frame_resized = cv2.resize(frame, (640, 480))
            frame_rgb_disp = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb_disp)
            imgtk = ImageTk.PhotoImage(image=img_pil)
            video_panel.imgtk = imgtk
            video_panel.configure(image=imgtk)
            video_panel.update_idletasks()
            video_panel.update()
        except Exception as e:
            print("Display error:", e)

    cap.release()
    video_panel.configure(image='')
    print("Camera loop exited.")

# =========================
# Camera control functions
# =========================
def start_camera():
    global cam_thread, cam_stop_event
    cam_stop_event.clear()
    start_cam_btn.config(state="disabled")
    stop_cam_btn.config(state="normal")
    cam_thread = Thread(target=camera_loop, daemon=True)
    cam_thread.start()

def stop_camera():
    cam_stop_event.set()
    start_cam_btn.config(state="normal")
    stop_cam_btn.config(state="disabled")

start_cam_btn.config(command=start_camera)
stop_cam_btn.config(command=stop_camera)

# =========================
# Calibration & threshold
# =========================
def calibrate():
    try:
        known_m = simpledialog.askfloat("Calibrate", "Place object at known distance (m):", minvalue=0.1, maxvalue=50.0)
        if known_m is None:
            return
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            messagebox.showerror("Calibrate", "Could not capture camera frame.")
            return
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if midas is None:
            # proxy calibration
            baseline_rel = 0.25
            calibration['scale'] = known_m * baseline_rel
            messagebox.showinfo("Calibrate", f"Calibration (proxy) set to {calibration['scale']:.4f}")
            return
        depth_map = estimate_depth_map(frame_rgb)
        h, w = depth_map.shape
        cx1, cy1 = int(w*0.35), int(h*0.35)
        cx2, cy2 = int(w*0.65), int(h*0.65)
        crop = depth_map[cy1:cy2, cx1:cx2]
        median_depth = float(np.median(crop))
        calibration['scale'] = known_m * median_depth
        messagebox.showinfo("Calibrate", f"Calibration complete. New scale set to {calibration['scale']:.4f}")
    except Exception as e:
        messagebox.showerror("Calibrate", str(e))

calib_btn.config(command=calibrate)

def set_threshold():
    val = simpledialog.askfloat("Set Threshold", "Enter alert threshold distance in meters:", minvalue=0.1, maxvalue=100.0)
    if val is None:
        return
    alert_threshold.set(val)
    threshold_label.config(text=f"Current alert threshold: {alert_threshold.get():.2f} m")

set_thresh_btn.config(command=set_threshold)

# =========================
# Cleanup & run
# =========================
def on_close():
    stop_camera()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)

if __name__ == "__main__":
    print("🚀 Starting Vision Assist...")
    if processor is None or blip_model is None:
        print("Note: BLIP captioning not available.")
    if midas is None or transform is None:
        print("Note: MiDaS depth estimation not available.")
    if yolo_model is None:
        print("Note: YOLO detection not available.")
    if tts_engine is None:
        print("Note: TTS engine not available.")
    root.mainloop()