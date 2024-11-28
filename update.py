# import tensorflow as tf
# print("TensorFlow version:", tf.__version__)
# print("Keras version:", tf.keras.__version__)

# TensorFlow version: 2.10.1
# Keras version: 2.10.0

# from tensorflow.keras.models import load_model
# from tensorflow.keras.metrics import MeanAbsoluteError  # Import lớp MeanAbsoluteError

# # Tải mô hình và sử dụng custom_objects với 'mae' là lớp MeanAbsoluteError
# model = load_model('F:\\LT_IT\\XuLyAnh\\wp\\BTL\\UTKFace\\age_gender_model.h5', custom_objects={'mae': MeanAbsoluteError()})

# # Kiểm tra mô hình đã được tải chưa
# print(model.summary())

############################################################################################################
import cv2
import numpy as np
from keras.models import load_model
from tensorflow.keras.utils import img_to_array
from PIL import Image
from tensorflow.keras.metrics import MeanAbsoluteError  # Import lớp MeanAbsoluteError

# Tải mô hình và sử dụng custom_objects với 'mae' là lớp MeanAbsoluteError
model = load_model('F:\\LT_IT\\XuLyAnh\\wp\\BTL\\UTKFace\\age_gender_model.h5', custom_objects={'mae': MeanAbsoluteError()})

# Define a dictionary for gender labels
gender_dict = {0: 'Male', 1: 'Female'}

# Function to predict age and gender
def predict_age_gender(image, model):
    # Preprocess the image
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (128, 128))
    img = img_to_array(img).reshape(1, 128, 128, 1) / 255.0

    # Make predictions
    pred = model.predict(img)
    pred_gender = gender_dict[round(pred[0][0][0])]
    pred_age = round(pred[1][0][0])

    return pred_gender, pred_age

# Function to process webcam feed
def webcam_prediction(model):
    # Load the face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image.")
            break

        # Flip the frame horizontally to simulate mirror effect
        frame = cv2.flip(frame, 1)

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Crop the face from the frame
            face = frame[y:y+h, x:x+w]

            # Predict age and gender
            gender, age = predict_age_gender(face, model)

            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Display the predictions on the frame
            cv2.putText(frame, f"Gender: {gender}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Age: {age}", (x, y+h+10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Age and Gender Prediction', frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to predict from an image file
def image_prediction(image_path, model):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image at {image_path}")
        return

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load the face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Crop the face from the image
        face = image[y:y+h, x:x+w]

        # Predict age and gender
        gender, age = predict_age_gender(face, model)

        # Draw rectangle around the face
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Display the predictions on the image
        cv2.putText(image, f"Gender: {gender}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(image, f"Age: {age}", (x, y+h+10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show the image
    cv2.imshow('Age and Gender Prediction', cv2.resize(image, (640, 480)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Main function to choose between image and webcam
if __name__ == "__main__":
    print("Choose an option:")
    print("1. Predict from webcam")
    print("2. Predict from an image file")
    choice = input("Enter 1 or 2: ")

    if choice == '1':
        webcam_prediction(model)
    elif choice == '2':
        image_path = input("Enter the image file path: ")
        image_prediction(image_path, model)
    else:
        print("Invalid choice!")

