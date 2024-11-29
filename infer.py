# -*- coding: utf-8 -*-
import os
from pathlib import Path
from openvino.runtime import Core
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def preprocess_image(image_path, input_shape):
    image = Image.open(image_path).convert("RGB")
    target_width = input_shape[3]
    target_height = input_shape[2]
    image = image.resize((target_width, target_height))
    image = np.array(image).astype(np.float32)
    image /= 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image

def visualize_results(original_image_path, anomaly_map, output_dir, threshold=0.3):
    original_image = Image.open(original_image_path).convert("RGB")
    anomaly_map_vis = (anomaly_map * 255).astype(np.uint8)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(anomaly_map_vis, cmap='jet')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / f"inference_result_{Path(original_image_path).stem}.png")
    plt.close()
    anomaly_score = anomaly_map.mean()
    is_anomalous = anomaly_score < threshold
    return is_anomalous

def main():
    ie = Core()
    model_xml = Path("Anomalie-Erkennung-Metal/MVtech/model.xml")
    model_bin = model_xml.with_suffix('.bin')
    metadata = model_xml.parent / "metadata.json"

    if not (model_xml.exists() and model_bin.exists() and metadata.exists()):
        exit(1)

    model = ie.read_model(model=model_xml, weights=model_bin)
    input_layer = model.inputs[0]
    output_layer = model.outputs[0]

    if not input_layer.partial_shape.is_static:
        new_shape = [1, 3, 700, 700]
        model.reshape({input_layer.any_name: new_shape})

    compiled_model = ie.compile_model(model=model, device_name="CPU")
    compiled_input = compiled_model.input(0)
    compiled_output = compiled_model.output(0)

    test_images_dir = Path("Anomalie-Erkennung-Metal/Tests")
    training_images_dir = Path("Anomalie-Erkennung-Metal/Trainingsbilder")

    if not test_images_dir.exists():
        exit(1)
    if not training_images_dir.exists():
        exit(1)

    output_dir = Path("Anomalie-Erkennung-Metal/inference_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    supported_formats = ['.png', '.jpg', '.jpeg', '.bmp']
    test_images = [img for img in test_images_dir.iterdir() if img.suffix.lower() in supported_formats]
    training_images = [img for img in training_images_dir.iterdir() if img.suffix.lower() in supported_formats] if training_images_dir.exists() else []

    all_images = test_images + training_images
    if not all_images:
        exit(1)

    for image_path in all_images:
        try:
            processed_image = preprocess_image(str(image_path), compiled_input.shape)
        except Exception:
            continue
        try:
            result = compiled_model([processed_image])[compiled_output]
        except Exception:
            continue
        try:
            anomaly_map = np.array(result).squeeze()
        except Exception:
            continue
        if anomaly_map.max() - anomaly_map.min() != 0:
            anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())
        else:
            anomaly_map = np.zeros_like(anomaly_map)
        visualize_results(str(image_path), anomaly_map, output_dir, threshold=0.22)

if __name__ == "__main__":
    main()
