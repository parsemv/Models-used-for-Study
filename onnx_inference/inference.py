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
    class_ids = []

    for pred in predictions[0]:
        conf = pred[4]
        if conf > conf_threshold:
            x, y, w, h = pred[0:4]
            x1 = int((x - w / 2) * original_shape[1] / 640)
            y1 = int((y - h / 2) * original_shape[0] / 640)
            x2 = int((x + w / 2) * original_shape[1] / 640)
            y2 = int((y + h / 2) * original_shape[0] / 640)
            cls_scores = pred[5:]
            class_id = np.argmax(cls_scores)
            boxes.append([x1, y1, x2, y2])
            scores.append(float(conf))
            class_ids.append(int(class_id))
    return boxes, scores, class_ids

# Run inference on an image
def run_inference(image_path):
    input_tensor, raw_img, orig_shape = preprocess_image(image_path)
    inputs = {session.get_inputs()[0].name: input_tensor}
    outputs = session.run(None, inputs)
    boxes, scores, class_ids = postprocess(outputs, orig_shape)

    # Draw results
    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box
        cv2.rectangle(raw_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Class {class_id} ({score:.2f})"
        cv2.putText(raw_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    output_path = "output.jpg"
    cv2.imwrite(output_path, raw_img)
    print(f"Inference complete. Output saved to {output_path}")

# Example usage
if __name__ == "__main__":
    test_image = "test.jpg"  # Replace with your test image path
    run_inference(test_image)
