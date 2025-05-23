import onnxruntime as ort
import numpy as np
import cv2
import os

# Load ONNX model
model_path = os.path.join(os.path.dirname(__file__), "best.onnx")
session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

# Image preprocessing
def preprocess_image(image_path, input_size=(640, 640)):
    img = cv2.imread(image_path)
    original_shape = img.shape[:2]
    img_resized = cv2.resize(img, input_size)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_transposed = np.transpose(img_rgb, (2, 0, 1)).astype(np.float32) / 255.0
    img_input = np.expand_dims(img_transposed, axis=0)
    return img_input, img, original_shape

# Postprocessing to extract boxes
def postprocess(predictions, original_shape, conf_threshold=0.25):
    boxes = []
    scores = []
