import time
import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
import tensorflow as tf
import numpy as np
import keras.utils as image

# Paths
model_path = 'model_unquant.tflite'
label_path = '/mnt/data/labels.txt'

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

# Main function
def main():
    ads = setup_ads1115()
    print("Detecting alcohol levels...")

    try:
        while True:
            alcohol_voltage = read_alcohol_level(ads)
            print(f"Alcohol Level: {alcohol_voltage:.3f} V")

            if alcohol_voltage > 0.4:
                print("Alcohol detected! Analyzing face...")
                # Capture and process image here (e.g., with OpenCV)
                img_path = 'path_to_captured_image.jpg'  # Example path
                pred_max, lbl_max = predict_face(model_path, label_path, img_path)
                print(f"Prediction: {lbl_max}, Confidence: {pred_max:.2f}")

            time.sleep(1)

    except KeyboardInterrupt:
        print("Measurement stopped by user.")

if __name__ == "__main__":
    main()
