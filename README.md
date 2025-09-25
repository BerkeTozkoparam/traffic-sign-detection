# YOLOv8 Traffic Sign Detection Dashboard

This project is a **real-time traffic sign detection application** built with **YOLOv8** and **Streamlit**. It allows users to upload images or use a webcam feed to detect common traffic signs like Stop, Speed Limit, and Traffic Lights.  

---

## Project Screenshot
<img width="672" height="679" alt="Ekran Resmi 2025-09-25 17 00 30" src="https://github.com/user-attachments/assets/433a761e-3e6a-41bd-bad7-2cbf145b288e" />



> Screenshot of the Streamlit dashboard with traffic sign detection in action.

---

## Features

- **Image Upload:** Test your model by uploading traffic images.  
- **Webcam Support:** Run real-time detection using your webcam.  
- **Bounding Boxes & Labels:** Shows detected traffic signs with confidence scores.  
- **Interactive Dashboard:**  
  - Detailed detection table  
  - Traffic sign statistics graph  
  - Warnings for critical signs like Stop  
- **Download Results:** Export detection results as CSV or JSON.  

---

## Technology Stack

- **Python 3.10+**  
- **Streamlit:** Interactive web interface  
- **Ultralytics YOLOv8:** Object detection model  
- **OpenCV & NumPy:** Image processing  
- **Pandas:** Data handling for detection results  
- **Matplotlib:** Graphical visualization of statistics  
- **Torch:** Deep learning backend for YOLOv8  

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/YourUsername/traffic-sign-detection.git
cd traffic-sign-detection
