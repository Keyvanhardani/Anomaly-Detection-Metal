
# Anomaly Detection for Metal Parts in Production 🏭🔧

This repository provides the code for anomaly detection in metal parts during production processes. Due to the large size of the pre-trained model, it is hosted on [Hugging Face](https://huggingface.co/Keyven/AnomalyDetection-MVTech-Metal/).

---

## Downloading the Model

The model files can be automatically downloaded using the Hugging Face Hub library. Run the following script to download all necessary files into the correct directory:

```python
from huggingface_hub import hf_hub_download
from pathlib import Path

def download_model():
    model_dir = Path("Anomalie-Erkennung-Metal/MVTech")
    model_dir.mkdir(parents=True, exist_ok=True)

    hf_hub_download(repo_id="Keyven/AnomalyDetection-MVTech-Metal", filename="model.xml", cache_dir=str(model_dir))
    hf_hub_download(repo_id="Keyven/AnomalyDetection-MVTech-Metal", filename="model.bin", cache_dir=str(model_dir))
    hf_hub_download(repo_id="Keyven/AnomalyDetection-MVTech-Metal", filename="metadata.json", cache_dir=str(model_dir))

if __name__ == "__main__":
    download_model()
```

Alternatively, you can manually download the files from [Hugging Face](https://huggingface.co/Keyven/AnomalyDetection-MVTech-Metal/) and place them in the directory `Anomalie-Erkennung-Metal/MVTech`.

---

## Installation

Make sure you have all required dependencies installed:
```bash
pip install -r requirements.txt
```

---

## Running the Inference Script

After downloading the model, you can run the inference script to analyze test images:

```bash
python infer.py
```

---

## Directory Structure

The project expects the following structure:

```
Anomalie-Erkennung-Metal/
│
├── MVTech/
│   ├── model.xml
│   ├── model.bin
│   ├── metadata.json
│
├── Tests/
│   └── [Test Images]
│
├── Trainingsbilder/
│   └── [Training Images]
│
├── inference_results/
│   └── [Generated Anomaly Maps]
│
├── infer.py
├── requirements.txt
├── download_model.py
└── README.md
```

---

## Notes

- **Model Hosting**: The model files are hosted on [Hugging Face](https://huggingface.co/Keyven/AnomalyDetection-MVTech-Metal/) due to size limitations.
- **Pre-trained Model**: This model was trained on the `mvtec-ad` dataset and uses OpenVINO for inference.
- **Performance Metrics**:
  - **Image AUROC**: 0.95
  - **Image F1 Score**: 0.94
  - **Pixel AUROC**: 0.96
  - **Pixel F1 Score**: 0.71

Feel free to report any issues or share feedback!
