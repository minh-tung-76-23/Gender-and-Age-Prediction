import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from keras.models import load_model
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.metrics import MeanAbsoluteError
from PIL import Image, ImageTk

# Tải mô hình và sử dụng custom_objects với 'mae' là lớp MeanAbsoluteError
model = load_model('F:\\LT_IT\\XuLyAnh\\wp\\BTL\\UTKFace\\age_gender_model_40.h5', custom_objects={'mae': MeanAbsoluteError()})

# Định nghĩa từ điển cho giới tính
gender_dict = {0: 'Male', 1: 'Female'}

# Hàm phân loại độ tuổi theo các nhóm
def age_group(age):
    if age <= 3:
        return "0-3"
    elif 4 <= age <= 7:
        return "4-7"
    elif 8 <= age <= 14:
        return "8-14"
    elif 15 <= age <= 24:
        return "15-24"
    elif 25 <= age <= 37:
        return "25-37"
    elif 38 <= age <= 47:
        return "38-47"
    elif 48 <= age <= 60:
        return "48-60"
    elif 61 <= age <= 100:
        return "61-100"
    elif 101 <= age <= 116:
        return "101-115"
    else:
        return "> 116"

# Hàm tiền xử lý ảnh chung
def preprocess_image(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (128, 128))
    return img_to_array(img).reshape(1, 128, 128, 1) / 255.0

# Hàm dự đoán giới tính và độ tuổi
def predict_age_gender(image, model):
    img = preprocess_image(image)
    pred = model.predict(img)
    pred_gender = gender_dict[round(pred[0][0][0])]
    pred_age = round(pred[1][0][0])
    return pred_gender, age_group(pred_age)

# Hàm tải bộ phát hiện khuôn mặt 
def load_face_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Hàm xử lý webcam
def webcam_prediction(model, window, label_result):
    face_cascade = load_face_cascade()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không thể chụp hình.")
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:  # Nếu phát hiện khuôn mặt
            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                gender, age_group = predict_age_gender(face, model)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, f"Gender: {gender}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Age: {age_group}", (x, y+h+10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Cập nhật kết quả lên giao diện
            label_result.config(text=f"Đã tắt Webcam")
        else:
            label_result.config(text="Không phát hiện khuôn mặt khi dùng webcam.")

        cv2.imshow('Webcam - Face, Age & Gender Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Hàm dự đoán từ ảnh
def image_prediction(image_path, model, label_result):
    image = cv2.imread(image_path)

    if image is None:
        print(f"Không thể tải ảnh tại {image_path}")
        return

    face_cascade = load_face_cascade()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:  # Nếu phát hiện khuôn mặt
        for (x, y, w, h) in faces:
            face = image[y:y+h, x:x+w]
            gender, age_group = predict_age_gender(face, model)
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(image, f"Gender: {gender}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(image, f"Age: {age_group}", (x, y+h+10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        label_result.config(text=f"Phát hiện khuôn mặt: {len(faces)}")
    else:  # Nếu không phát hiện khuôn mặt, dự đoán cho cả ảnh
        label_result.config(text="Không phát hiện khuôn mặt, dự đoán cho toàn bộ ảnh.")
        gender, age_group = predict_age_gender(image, model)
        cv2.putText(image, f"Gender: {gender}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(image, f"Age: {age_group}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Hiển thị ảnh với kích thước gốc
    cv2.imshow('Image - Face, Age & Gender Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Hàm mở hộp thoại chọn ảnh
def open_image(window, model, label_result):
    file_path = filedialog.askopenfilename(title="Chọn ảnh", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        image_prediction(file_path, model, label_result)

# Hàm main
def main():
    window = tk.Tk()
    window.title("Dự đoán Giới tính và Tuổi")

    # Đặt chiều rộng và chiều cao của form
    window_width = 500
    window_height = 200

    # Lấy kích thước màn hình
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()

    # Tính toán vị trí để căn giữa form
    x_cordinate = int((screen_width / 2) - ((window_width / 2)-100))
    y_cordinate = int((screen_height / 2) - (window_height / 2))

    # Đặt kích thước và vị trí của cửa sổ
    window.geometry(f"{window_width}x{window_height}+{x_cordinate}+{y_cordinate}")

    # Tiêu đề
    label_title = tk.Label(window, text="Nhận diện tuổi và giới tính", font=("Helvetica", 16))
    label_title.pack(pady=10)

    global label_image
    label_image = tk.Label(window)
    label_image.pack()

    label_result = tk.Label(window, text="Chọn chức năng", font=("Helvetica", 10))
    label_result.pack(pady=10)

    # Các nút bấm
    btn_open_image = tk.Button(window, text="Chọn ảnh", command=lambda: open_image(window, model, label_result))
    btn_open_image.pack(side=tk.LEFT, padx=10, pady=20)

    btn_open_webcam = tk.Button(window, text="Mở webcam", command=lambda: webcam_prediction(model, window, label_result))
    btn_open_webcam.pack(side=tk.RIGHT, padx=10, pady=20)

    # Bắt đầu vòng lặp
    window.mainloop()


if __name__ == "__main__":
    main()
