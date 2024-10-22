import time
import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
import tensorflow as tf
import numpy as np
import keras.utils as image
from picamera2 import Picamera2
import cv2
import firebase_admin
from firebase_admin import credentials, firestore, storage
import os

# Paths
model_path = 'model_unquant.tflite'
label_path = '/mnt/data/labels.txt'
image_path = '/tmp/captured_image.jpg'  # Temporary location to save the captured image
firebase_creds_path = 'path/to/firebase/credentials.json'  # Change this to your Firebase credentials path

# Initialize Firebase
cred = credentials.Certificate(firebase_creds_path)
firebase_admin.initialize_app(cred, {
    'storageBucket': 'your-firebase-storage-bucket-url.appspot.com'
})
db = firestore.client()
bucket = storage.bucket()

# ADS1115 setup
def setup_ads1115():
    i2c = busio.I2C(board.SCL, board.SDA)
    ads = ADS.ADS1115(i2c)
    return ads

def read_alcohol_level(ads, channel=0):
    chan = AnalogIn(ads, getattr(ADS, f"P{channel}"))
    return chan.voltage

# TFLite Model Setup
def load_tflite_model(model_path):
    tflite_interpreter = tf.lite.Interpreter(model_path=model_path)
    tflite_interpreter.allocate_tensors()
    input_details = tflite_interpreter.get_input_details()
    output_details = tflite_interpreter.get_output_details()
    return tflite_interpreter, input_details, output_details

def load_labels(label_path):
    with open(label_path, 'r') as f:
        labels = list(map(str.strip, f.readlines()))
    return labels

def load_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img = tf.expand_dims(img_array, 0)  # Create a batch
    img = np.array(img, dtype=np.float32)
    return img

def predict_face(model_path, label_path, img_path):
    tflite_interpreter, input_details, output_details = load_tflite_model(model_path)
    labels = load_labels(label_path)
    img = load_image(img_path)
    tflite_interpreter.set_tensor(input_details[0]['index'], img)

    # Run inference
    tflite_interpreter.invoke()
    predictions = tflite_interpreter.get_tensor(output_details[0]['index'])[0]
    top_k_indices = np.argsort(predictions)[::-1][:len(labels)]

    pred_max = predictions[top_k_indices[0]] / 255.0
    lbl_max = labels[top_k_indices[0]]
    return pred_max, lbl_max

# Capture an image using Picamera2
def capture_image(image_path):
    picam2 = Picamera2()
    picam2.configure(picam2.create_still_configuration())
    picam2.start()
    time.sleep(2)  # Allow time for the camera to warm up
    picam2.capture_file(image_path)
    picam2.stop()
    print(f"Image captured and saved to {image_path}")

# Check if a face is present using OpenCV
def check_face(image_path):
    img = cv2.imread(image_path)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return len(faces) > 0

# Upload image to Firebase Storage and get the download URL
def upload_image_to_firebase(image_path):
    blob = bucket.blob(os.path.basename(image_path))
    blob.upload_from_filename(image_path)
    return blob.public_url

# Store results in Firestore
def store_in_firestore(alcohol_level, confidence, label, image_url):
    doc_ref = db.collection('users').document('user_id').collection('tests').document('MM-dd-yyyy')
    timestamp = time.strftime("%H:%M:%S", time.localtime())
    test_data = {
        "time": timestamp,
        "alcohol_level": alcohol_level,
        "confidence": confidence,
        "label": label,
        "image_url": image_url,
        "verbal_test": ""  # Placeholder for future verbal test implementation
    }
    doc_ref.update({f"test_{timestamp}": test_data})

# Main function
def main():
    ads = setup_ads1115()
    print("Detecting alcohol levels...")

    try:
        while True:
            alcohol_voltage = read_alcohol_level(ads)
            print(f"Alcohol Level: {alcohol_voltage:.3f} V")

            if alcohol_voltage > 0.4:
                print("Alcohol detected! Capturing image...")
                capture_image(image_path)

                # Ensure the user is looking at the camera
                if check_face(image_path):
                    print("Face detected, analyzing...")
                    pred_max, lbl_max = predict_face(model_path, label_path, image_path)
                    print(f"Prediction: {lbl_max}, Confidence: {pred_max:.2f}")

                    # Upload the image to Firebase Storage
                    image_url = upload_image_to_firebase(image_path)

                    # Store the data in Firestore
                    store_in_firestore(alcohol_voltage, pred_max, lbl_max, image_url)
                else:
                    print("No face detected. Please ensure you're looking at the camera.")

            time.sleep(1)

    except KeyboardInterrupt:
        print("Measurement stopped by user.")

if __name__ == "__main__":
    main()
