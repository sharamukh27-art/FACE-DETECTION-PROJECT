import argparse
import os
import time

import cv2
import numpy as np
import tensorflow as tf


CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def get_model_input_details(model):
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    height = input_shape[1] or 96
    width = input_shape[2] or 96
    channels = input_shape[3] or 3
    return int(width), int(height), int(channels)


def load_emotion_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = tf.keras.models.load_model(model_path, compile=False)
    width, height, channels = get_model_input_details(model)
    print(f"Loaded model: {model_path}")
    print(f"Model input: {height}x{width}x{channels}")
    return model, (width, height, channels)


def find_largest_face(face_cascade, frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(60, 60),
    )

    if len(faces) == 0:
       return None
    return max(faces, key=lambda box: box[2] * box[3])


def preprocess_face(frame, face_box, input_details):
    width, height, channels = input_details
    x, y, w, h = face_box

    pad = int(0.15 * max(w, h))
    x1 = max(x - pad, 0)
    y1 = max(y - pad, 0)
    x2 = min(x + w + pad, frame.shape[1])
    y2 = min(y + h + pad, frame.shape[0])

    face = frame[y1:y2, x1:x2]

    if channels == 1:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (width, height))
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=-1)
    else:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (width, height))
        face = face.astype("float32") / 255.0

    return np.expand_dims(face, axis=0)


def draw_prediction(frame, face_box, label, confidence):
    x, y, w, h = face_box
    text = f"{label}: {confidence:.2f}"

    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.rectangle(frame, (x, max(y - 34, 0)), (x + 230, y), (0, 255, 0), -1)
    cv2.putText(
        frame,
        text,
        (x + 8, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )


def run_webcam(model_path, camera_index):
    model, input_details = load_emotion_model(model_path)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if face_cascade.empty():
        raise RuntimeError("Could not load OpenCV Haar cascade for face detection.")

    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  
    cap.set(cv2.CAP_PROP_FPS, 30)                                  
    time.sleep(2)                                                   
    for _ in range(10):                                             
        cap.read()       
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {camera_index}.")

    print("Webcam started. Press 'q' to quit.")

    try:
        while True:
            ok, frame = cap.read()  
            if not ok:
                print("Could not read frame from webcam.")
                break

            face_box = find_largest_face(face_cascade, frame)

            if face_box is not None:
                processed = preprocess_face(frame, face_box, input_details)
                predictions = model.predict(processed, verbose=0)[0]
                class_index = int(np.argmax(predictions))
                label = CLASS_NAMES[class_index]
                confidence = float(predictions[class_index])
                draw_prediction(frame, face_box, label, confidence)
            else:
                cv2.putText(
                    frame,
                    "No face detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Emotion Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Test a saved emotion model with webcam.")
    parser.add_argument(
        "--model",
        default="/Users/sharanyamukherjee55/Downloads/somth/trained_emotion_model.keras",
        help="Path to the saved Keras model.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam index. Try 1 if 0 does not work.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_webcam(args.model, args.camera)

