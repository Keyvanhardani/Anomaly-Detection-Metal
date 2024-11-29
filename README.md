# Anomaly Detection for Metal Parts in Production 🏭🔧

This repository demonstrates an anomaly detection system for identifying defects in metal parts during production lines using **OpenVINO** and **Deep Learning**. The model detects anomalies such as scratches or deformations in real time.

---

## Features
- **Image Preprocessing**: Automatically prepares input images for the model.
- **Anomaly Detection**: Generates heatmaps and segmentation masks to highlight defects.
- **Visualization**: Saves output images and anomaly maps for further inspection.

---

## Performance Metrics
- **Image AUROC**: 0.95
- **Image F1 Score**: 0.94
- **Pixel AUROC**: 0.96
- **Pixel F1 Score**: 0.71

---

## Installation
To set up the environment, clone the repository and install the dependencies:
```bash
git clone https://github.com/Keyvanhardani/Anomaly-Detection-Metal/
cd Anomaly-Detection-Metal
pip install -r requirements.txt
