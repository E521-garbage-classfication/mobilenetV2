import os
import random
import shutil

# 定義資料集路徑
original_data_dir = r'C:\E521C\DataImage\trash_dataset_v2_new'  # 原始資料集路徑
output_dir = r'C:\E521C\DataImage\trash_dataset_v2_split'  # 用於存放分割後資料的根目錄

# 定義訓練、驗證、測試集的比例
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# 檢查並創建輸出資料夾
os.makedirs(output_dir, exist_ok = True)

# 創建訓練、驗證、測試資料夾
train_dir = os.path.join(output_dir, 'train')
val_dir = os.path.join(output_dir, 'val')
test_dir = os.path.join(output_dir, 'test')

os.makedirs(train_dir, exist_ok = True)
os.makedirs(val_dir, exist_ok = True)
os.makedirs(test_dir, exist_ok = True)

# 創建每個類別的資料夾
class_names = [folder for folder in os.listdir(original_data_dir) if os.path.isdir(os.path.join(original_data_dir, folder))]
for class_name in class_names:
    os.makedirs(os.path.join(train_dir, class_name), exist_ok = True)
    os.makedirs(os.path.join(val_dir, class_name), exist_ok = True)
    os.makedirs(os.path.join(test_dir, class_name), exist_ok = True)

# 分割資料集：將每個類別的圖片分成訓練集、驗證集、測試集
for class_name in class_names:
    class_dir = os.path.join(original_data_dir, class_name)
    images = [img for img in os.listdir(class_dir) if img.lower().endswith(('png', 'jpg', 'jpeg'))]
    
    # 隨機打亂圖片順序
    random.shuffle(images)
    
    # 計算每個子集的大小
    total_images = len(images)
    train_size = int(total_images * train_ratio)
    val_size = int(total_images * val_ratio)
    test_size = total_images - train_size - val_size  # 剩下的就是測試集
    
    # 分配資料
    train_images = images[:train_size]
    val_images = images[train_size:train_size + val_size]
    test_images = images[train_size + val_size:]
    
    # 複製圖片到對應的資料夾
    for img in train_images:
        src = os.path.join(class_dir, img)
        dst = os.path.join(train_dir, class_name, img)
        shutil.copy(src, dst)
    
    for img in val_images:
        src = os.path.join(class_dir, img)
        dst = os.path.join(val_dir, class_name, img)
        shutil.copy(src, dst)
    
    for img in test_images:
        src = os.path.join(class_dir, img)
        dst = os.path.join(test_dir, class_name, img)
        shutil.copy(src, dst)

print(f"Saved at {output_dir}")
