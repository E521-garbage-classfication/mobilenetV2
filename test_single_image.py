import tensorflow as tf
from tensorflow.keras.applications import mobilenet_v2
import numpy as np
from PIL import Image
import sys

# 模型路徑
model_path = r'C:\E521C\saved_models_v2_new\trash_model_advanced.h5'

# 你訓練時的類別順序，照 train_ds.class_names 排序填寫
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic']

def load_and_preprocess_image(img_path, target_size=(192, 192)):
    img = Image.open(img_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)  # 增加 batch 維度
    return img_array

def predict_image(img_path):
    model = tf.keras.models.load_model(model_path)
    img_tensor = load_and_preprocess_image(img_path)
    preds = model.predict(img_tensor)
    pred_class_idx = np.argmax(preds)
    pred_class = class_names[pred_class_idx]
    pred_prob = preds[0][pred_class_idx]
    print(f"Prediction: {pred_class} ({pred_prob*100:.2f}%)")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python test_single_image.py <image_path>")
        sys.exit(1)
    img_path = sys.argv[1]
    predict_image(img_path)
