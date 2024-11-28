import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from tqdm import tqdm
from PIL import Image
import tensorflow as tf
from keras.preprocessing.image import load_img
from keras.models import Model
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Input
from tensorflow.keras.utils import plot_model

warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (10, 6)

# Đường dẫn tập dữ liệu (sửa thành đường dẫn trên máy của bạn)
BASE_DIR = "F:\\LT_IT\\XuLyAnh\\wp\\BTL\\dataset\\UTK_face\\UTKFace"

# Bước 1: Load dữ liệu
image_paths, age_labels, gender_labels = [], [], []

for filename in tqdm(os.listdir(BASE_DIR)):
    image_path = os.path.join(BASE_DIR, filename)
    temp = filename.split('_')
    age = int(temp[0])
    gender = int(temp[1])
    image_paths.append(image_path)
    age_labels.append(age)
    gender_labels.append(gender)

# Chuyển dữ liệu vào DataFrame
df = pd.DataFrame({'image': image_paths, 'age': age_labels, 'gender': gender_labels})

# Map nhãn giới tính
gender_dict = {0: 'Male', 1: 'Female'}

# Hiển thị phân phối tuổi và giới tính
# sns.distplot(df['age'], kde=False, bins=30)
# plt.title("Age Distribution")
# plt.show()

# sns.countplot(df['gender'])
# plt.title("Gender Distribution")
# plt.show()

# Bước 2: Hiển thị lưới ảnh mẫu
plt.figure(figsize=(20, 20))
files = df.iloc[:25]
for idx, (file, age, gender) in enumerate(zip(files['image'], files['age'], files['gender'])):
    plt.subplot(5, 5, idx + 1)
    img = load_img(file)
    plt.imshow(img)
    plt.title(f"Age: {age}, Gender: {gender_dict[gender]}")
    plt.axis('off')
plt.show()

# Bước 3: Xử lý ảnh
def extract_features(images):
    features = []
    for image in tqdm(images):
        img = load_img(image, color_mode='grayscale')
        img = img.resize((128, 128), Image.Resampling.LANCZOS)
        features.append(np.array(img))
    return np.array(features).reshape(len(features), 128, 128, 1)

# Trích xuất đặc trưng
X = extract_features(df['image'])
X = X / 255.0  # Chuẩn hóa ảnh

y_gender = np.array(df['gender'])
y_age = np.array(df['age'])

# Bước 4: Xây dựng mô hình
input_shape = (128, 128, 1)
inputs = Input(input_shape)

conv_1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)
maxp_1 = MaxPooling2D(pool_size=(2, 2))(conv_1)
conv_2 = Conv2D(64, kernel_size=(3, 3), activation='relu')(maxp_1)
maxp_2 = MaxPooling2D(pool_size=(2, 2))(conv_2)
conv_3 = Conv2D(128, kernel_size=(3, 3), activation='relu')(maxp_2)
maxp_3 = MaxPooling2D(pool_size=(2, 2))(conv_3)
conv_4 = Conv2D(256, kernel_size=(3, 3), activation='relu')(maxp_3)
maxp_4 = MaxPooling2D(pool_size=(2, 2))(conv_4)

flatten = Flatten()(maxp_4)

dense_1 = Dense(256, activation='relu')(flatten)
dense_2 = Dense(256, activation='relu')(flatten)

dropout_1 = Dropout(0.4)(dense_1)
dropout_2 = Dropout(0.4)(dense_2)

output_1 = Dense(1, activation='sigmoid', name='gender_out')(dropout_1)
output_2 = Dense(1, activation='relu', name='age_out')(dropout_2)

model = Model(inputs=[inputs], outputs=[output_1, output_2])

model.compile(loss=['binary_crossentropy', 'mae'], optimizer='adam', metrics=['accuracy'])

# Hiển thị cấu trúc mô hình
# plot_model(model, show_shapes=True, to_file="model_structure.png")

# Bước 5: Train mô hình
history = model.fit(x=X, y=[y_gender, y_age], batch_size=32, epochs=25, validation_split=0.2)

# Lưu mô hình
model.save("age_gender_model.h5")

# Bước 6: Đánh giá và vẽ đồ thị
best_gender_accuracy = max(history.history['gender_out_accuracy'])
best_val_gender_accuracy = max(history.history['val_gender_out_accuracy'])

print(f"Best Training Accuracy (Gender): {best_gender_accuracy:.2f}")
print(f"Best Validation Accuracy (Gender): {best_val_gender_accuracy:.2f}")

# Accuracy Graph
plt.plot(history.history['gender_out_accuracy'], label='Train Accuracy')
plt.plot(history.history['val_gender_out_accuracy'], label='Validation Accuracy')
plt.title("Gender Accuracy")
plt.legend()
plt.show()

# Loss Graph for Age
plt.plot(history.history['age_out_mae'], label='Train MAE')
plt.plot(history.history['val_age_out_mae'], label='Validation MAE')
plt.title("Age MAE")
plt.legend()
plt.show()

# Bước 7: Kiểm tra mô hình với một ảnh
image_index = 1010
print(f"Actual Gender: {gender_dict[y_gender[image_index]]}, Actual Age: {y_age[image_index]}")

pred = model.predict(X[image_index].reshape(1, 128, 128, 1))
pred_gender = gender_dict[round(pred[0][0][0])]
pred_age = round(pred[1][0][0])

print(f"Predicted Gender: {pred_gender}, Predicted Age: {pred_age}")
plt.imshow(X[image_index].reshape(128, 128), cmap='gray')
plt.axis('off')
plt.show()
