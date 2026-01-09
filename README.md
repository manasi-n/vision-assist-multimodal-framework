# Real-Time Multimodal Assistive Framework for Visually Impaired Users

## 📌 Overview
This project presents a **software-based real-time assistive system** designed to support visually impaired users by combining **computer vision, depth estimation, speech alerts, and multilingual image captioning**.

The framework is modular and extensible, allowing **future integration with embedded sensors** such as **ultrasonic, IR, or LiDAR** for robotic or wearable platforms.

---

## 🚀 Key Features
- 🎯 **Real-time object detection** using YOLOv8
- 📏 **Proximity estimation** using MiDaS depth estimation (with fallback methods)
- 🔊 **Voice-based alerts** for nearby obstacles
- 🖼️ **Image captioning** using BLIP (Vision–Language Model)
- 🌐 **Multilingual translation** (English → Telugu & Hindi)
- 🎥 Live camera feed with object tracking (IoU-based)
- 🧩 Modular architecture for future embedded integration
## 📸 Project Screenshots

### Live Object Detection & Distance Alerts
![Live Detection](assets/screenshot1.png)

### Image Captioning & Translation
![Captioning](assets/screenshot2.png)

### GUI Interface
![GUI](assets/screenshot3.png)

---

## 🧠 Technologies Used
- Python
- OpenCV
- YOLOv8 (Ultralytics)
- MiDaS (Monocular Depth Estimation)
- BLIP Image Captioning
- PyTorch
- Tkinter (GUI)
- pyttsx3 (Text-to-Speech)

---

## 🖥️ System Architecture
