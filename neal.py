import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # GPU 記憶體可動態成長
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'          # 關閉 TensorFlow 警告訊息
import glob
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from datetime import datetime
import sys
sys.stdout.flush()

# GPU 環境偵測與提示
# 若未偵測到 GPU 則會自動改用 CPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("✅ GPU is available and being used:", gpus)
else:
    print("❌ No GPU detected. Using CPU.")

# 訓練與儲存相關資料夾設定
model_dir = r'C:\E521C\saved_models_v2_new'
os.makedirs(model_dir, exist_ok = True)  # 若資料夾不存在則建立
count_path = os.path.join(model_dir, 'train_count.txt')  # 紀錄訓練次數的檔案

# 訓練次數累加邏輯
if os.path.exists(count_path):
    with open(count_path, 'r') as f:
        count = int(f.read().strip()) + 1
else:
    count = 1
with open(count_path, 'w') as f:
    f.write(str(count))

print(f"🧠 Training No.{count}", flush = True)

# 載入資料集
# 自資料夾中自動依照子資料夾分類進行標記，並切分為訓練集與驗證集
data_dir = r'C:\E521C\DataImage\trash_dataset_v2_new'
class_names = sorted([folder for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))])
num_classes = len(class_names)

# 使用 MobileNetV2 的預處理函數進行資料增強及預處理
train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split = 0.2,  # 驗證集比例為 20%
    subset = "both",
    seed = 123,              # 隨機種子，確保每次切割一致
    image_size = (128, 128),
    batch_size = 16,         # 訓練每次批量大小
    label_mode='int',
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input  # 使用預處理函數
)

# 影像增強（訓練集用）
augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),     # 隨機左右翻轉
    layers.RandomRotation(0.2),          # 隨機旋轉 ±20%
    layers.RandomZoom(0.2),              # 隨機放大/縮小
    layers.RandomContrast(0.1),          # 隨機對比度 ±10%
    layers.RandomTranslation(0.1, 0.1)   # 隨機平移
])

# 套用預處理流程：訓練集使用增強，驗證集不使用增強
train_ds = train_ds.map(lambda x, y: (augmentation(x), y))  # 只對訓練集進行增強

# 加入預抓機制（加快資料加載）
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size = AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size = AUTOTUNE)

# 模型儲存路徑
h5_path = os.path.join(model_dir, 'trash_model_advanced.h5')
keras_path = os.path.join(model_dir, 'trash_model_advanced.keras')

# 建立模型函數（可重複使用）
def build_model():
    base_model = MobileNetV2(input_shape = (128, 128, 3), include_top = False, weights = 'imagenet')  # 使用預訓練權重
    base_model.trainable = True
    for layer in base_model.layers[:100]:      # 凍結前 100 層，只微調最後幾層
        layer.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation = 'relu', kernel_regularizer = regularizers.l2(1e-4)),  # L2 正則化，避免過擬合
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation = 'relu', kernel_regularizer = regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation = 'softmax')  
    ])
    return model

# 載入模型（若有舊模型可繼續訓練）
try:
    if os.path.exists(h5_path):
        print("✅ Found .h5 model, continue training...", flush = True)
        model = load_model(h5_path)
    elif os.path.exists(keras_path):
        print("🧪 Found .keras model, converting to .h5...", flush = True)
        model = load_model(keras_path)
        model.save(h5_path)
        print("✅ Conversion complete, using .h5 model.", flush = True)
    else:
        raise FileNotFoundError
except Exception as e:
    print("❌ Error loading model:", e, flush = True)
    print("🔃 Creating new model...", flush = True)
    model = build_model()

# ✅ 編譯模型：指定 optimizer、loss 函數與評估指標
model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4),  # 初始學習率
    loss = 'sparse_categorical_crossentropy',  # 搭配整數標籤使用
    metrics = ['accuracy']
)

# 訓練模型（EarlyStopping + ReduceLROnPlateau 控制訓練）
early_stop = EarlyStopping(monitor = 'val_accuracy', patience = 5, restore_best_weights = True)  
reduce_lr = ReduceLROnPlateau(monitor = 'val_accuracy', factor = 0.5, patience = 3, min_lr = 1e-6, verbose = 1)  # 降低學習率
print("Start training", flush = True)
history = model.fit(train_ds, validation_data = val_ds, epochs = 50, callbacks = [early_stop, reduce_lr], verbose = 2)
print("End of training", flush = True)

# （Fine-tuning）：解凍更多層做進一步學習
# 第二階段微調
print(" Start fine-tuning phase 2 (unfreeze all layers)...", flush = True)
# 第二階段 callbacks：避免沿用前一階段的 callback 狀態
early_stop_fine = EarlyStopping(monitor = 'val_accuracy', patience = 5, restore_best_weights = True)
reduce_lr_fine = ReduceLROnPlateau(monitor = 'val_accuracy', factor = 0.5, patience = 3, min_lr = 1e-6, verbose = 1)

model.layers[0].trainable = True  

# 重新編譯模型
model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-5),  # 更小學習率
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

# 執行第二輪訓練
history_fine = model.fit(train_ds, validation_data = val_ds, epochs = 10, callbacks = [early_stop_fine, reduce_lr_fine], verbose = 2)

# 合併第二階段訓練紀錄
for key in history.history:
    history.history[key].extend(history_fine.history[key])

print("✅ Fine-tuning complete.", flush = True)

# 儲存模型（主模型與備份模型）
model.save(h5_path)
backup_model_path = os.path.join(model_dir, f'best_model_run{count}.h5')
model.save(backup_model_path)

# 保留最近 3 個模型，其餘自動刪除（節省空間）
all_models = sorted(glob.glob(os.path.join(model_dir, "best_model_run*.h5")), key = os.path.getmtime)
if len(all_models) > 3:
    for old_model in all_models[:-3]:
        os.remove(old_model)

# 繪製準確率曲線圖（訓練 vs 驗證）
plt.plot(history.history['accuracy'], label = 'Train Acc')
plt.plot(history.history['val_accuracy'], label = 'Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title(f'Training vs Validation Accuracy - Run {count}')
plt.legend()
acc_plot_path = os.path.join(model_dir, f'train_plot_run{count}.png')
plt.savefig(acc_plot_path)
plt.show()

# 繪製損失值曲線圖（訓練 vs 驗證）
plt.plot(history.history['loss'], label = 'Train Loss')
plt.plot(history.history['val_loss'], label = 'Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'Training vs Validation Loss - Run {count}')
plt.legend()
loss_plot_path = os.path.join(model_dir, f'train_loss_run{count}.png')
plt.savefig(loss_plot_path)
plt.show()

# 圖表過多時自動清理，只保留 15 張
# acc_plots = sorted(glob.glob(os.path.join(model_dir, "train_plot_run*.png")), key = os.path.getmtime)
# if len(acc_plots) > 15:
#     for old_plot in acc_plots[:-15]:
#         os.remove(old_plot)

# loss_plots = sorted(glob.glob(os.path.join(model_dir, "train_loss_run*.png")), key = os.path.getmtime)
# if len(loss_plots) > 15:
#     for old_plot in loss_plots[:-15]:
#         os.remove(old_plot)

# ✅ 訓練記錄寫入 log 檔案
log_text = f"Run {count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
log_text += f"Train Acc: {history.history['accuracy'][-1]:.4f} | Val Acc: {history.history['val_accuracy'][-1]:.4f}\n"
log_text += f"Train Loss: {history.history['loss'][-1]:.4f} | Val Loss: {history.history['val_loss'][-1]:.4f}\n"
log_text += f"Model Saved: {h5_path} and {backup_model_path}\n"
log_text += "-------------------------\n"

with open(os.path.join(model_dir, 'log.txt'), 'a', encoding = 'utf-8') as f:
    f.write(log_text)
