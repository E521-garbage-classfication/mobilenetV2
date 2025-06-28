import os
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import MobileNetV2, mobilenet_v2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all, 1 = info, 2 = warning, 3 = error only
tf.get_logger().setLevel('ERROR')
import matplotlib.pyplot as plt
import numpy as np
import logging
from datetime import datetime
# import json
import sys
sys.stdout.flush()
print(tf.__version__)
# model saved
model_dir = r'C:\E521C\saved_models_v2_new'
os.makedirs(model_dir, exist_ok = True)
count_path = os.path.join(model_dir, 'train_count.txt')

# train count
if os.path.exists(count_path):
    with open(count_path, 'r') as f:
        count = int(f.read().strip()) + 1
else:
    count = 1
with open(count_path, 'w') as f:
    f.write(str(count))
print(f"🧠 Training No.{count}", flush = True)

# dataset
train_dir = r"C:\E521C\DataImage\trash_dataset_v2_new_split\train"
val_dir   = r"C:\E521C\DataImage\trash_dataset_v2_new_split\val"
test_dir  = r"C:\E521C\DataImage\trash_dataset_v2_new_split\test"

BS = 16 # batch_size
size = 192  # image_size
IS = (size, size)

logging.getLogger("tensorflow").setLevel(logging.ERROR)
# 載入資料集
train_ds = tf.keras.utils.image_dataset_from_directory(train_dir, image_size = (192, 192), batch_size = BS, label_mode = 'categorical')
val_ds   = tf.keras.utils.image_dataset_from_directory(val_dir, image_size = (192, 192), batch_size = BS, label_mode = 'categorical')
test_ds  = tf.keras.utils.image_dataset_from_directory(test_dir, image_size = (192, 192), batch_size = BS, label_mode = 'categorical')

# 自訂顯示訊息
def count_images(directory):
    return sum([len(files) for _, _, files in os.walk(directory) if any(f.lower().endswith(('jpg','png','jpeg')) for f in files)])

num_train = count_images(train_dir)
num_val   = count_images(val_dir)
num_test  = count_images(test_dir)
num_classes = len(train_ds.class_names)

print(f"📂 Train: {num_train} files | {num_classes} classes")
print(f"🧪 Val: {num_val} files | {num_classes} classes")
print(f"🧾 Test: {num_test} files | {num_classes} classes")

# 類別數與名稱
num_classes = len(train_ds.class_names)
class_names = train_ds.class_names
print(f"Number of classes: {num_classes}")
print(class_names)

# 計算類別權重
# （需將 one-hot label 還原成 class index）
y_train_all = []
for _, labels in train_ds.unbatch():
    y_train_all.append(np.argmax(labels.numpy()))  # ← 取最大值 index
y_train_all = np.array(y_train_all)

class_weights_array = compute_class_weight(
    class_weight = 'balanced',
    classes = np.unique(y_train_all),
    y = y_train_all
)
class_weights = dict(enumerate(class_weights_array))
print("⚖️ Computed class weights:", class_weights)

# 前處理與增強
augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),   # horizontal_and_vertical
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.GaussianNoise(0.1),  
    layers.RandomBrightness(0.2),
    layers.RandomContrast(0.2),
    layers.RandomTranslation(0.1, 0.1)
])
def preprocess(x, y):
    x = mobilenet_v2.preprocess_input(x)
    return x, y
    
# ✅ 建立新的 augmentation + preprocess 合併函式
def preprocess_with_aug(x, y):
    x = augmentation(x)                         # 資料增強先執行
    x = mobilenet_v2.preprocess_input(x)        # 再進行 MobileNetV2 標準化
    return x, y

# ✅ 正確套用順序
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(preprocess_with_aug).prefetch(buffer_size = AUTOTUNE)
val_ds   = val_ds.map(preprocess).prefetch(buffer_size = AUTOTUNE)
test_ds  = test_ds.map(preprocess).prefetch(buffer_size = AUTOTUNE)

# 模型路徑
h5_path    = os.path.join(model_dir, 'trash_model_advanced.h5')
keras_path = os.path.join(model_dir, 'trash_model_advanced.keras')

def build_model():
    base_model = MobileNetV2(input_shape = (192, 192, 3), include_top = False, weights = 'imagenet')
    for layer in base_model.layers:
        layer.trainable = False

    model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation = 'relu', kernel_regularizer = regularizers.l2(1e-3)),  
    layers.BatchNormalization(),
    layers.Dropout(0.6),  
    layers.Dense(num_classes, activation = 'softmax', kernel_regularizer = regularizers.l2(1e-3))
])
    return model

# 進度條設定
class RichBatchProgressBar(Callback):
    def on_epoch_begin(self, epoch, logs = None):
        _ = logs
        steps = self.params.get('steps', 100)
        self.progress = Progress(
            TextColumn(f"[italic blue]Epoch {epoch + 1}/{self.params['epochs']}"),
            BarColumn(bar_width = None, complete_style = "bright_green", finished_style = "bright_green", pulse_style = "green"),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            transient = False  
        )
        self.task = self.progress.add_task("Training", total = steps)
        self.progress.start()

    def on_train_batch_end(self, batch, logs = None):
        _ = batch, logs
        self.progress.advance(self.task)

    def on_epoch_end(self, epoch, logs = None):
        _ = epoch, logs
        self.progress.stop()

try:
    if os.path.exists(h5_path):
        print("✅ Load existing .h5 model")
        model = load_model(h5_path)
    elif os.path.exists(keras_path):
        print("🧪 Convert .keras → .h5")
        model = load_model(keras_path)
        model.save(h5_path)
    else:
        raise FileNotFoundError
except Exception as e:
    print("🔃 Build new model:", e)
    model = build_model()

model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4),
    loss = CategoricalCrossentropy(from_logits = False, label_smoothing = 0.1),  # label_smoothing = 0.1
    metrics = ['accuracy']
)

early_stop_phase1 = EarlyStopping(monitor = 'val_loss', patience = 5, restore_best_weights = True, verbose = 1)
early_stop_phase2 = EarlyStopping(monitor = 'val_loss', patience = 5, restore_best_weights = True, verbose = 1)
reduce_lr_phase1 = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 3, min_lr = 1e-6, verbose = 1)
reduce_lr_phase2 = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.3, patience = 3, min_lr = 1e-7, verbose = 1)

checkpoint_phase1 = ModelCheckpoint(
    filepath = os.path.join(model_dir, f'best_val_acc_phase1_run{count}.h5'),
    monitor = 'val_loss',   #
    save_best_only = True,
    verbose = 1
)
checkpoint_phase2 = ModelCheckpoint(
    filepath = os.path.join(model_dir, f'best_val_acc_phase2_run{count}.h5'),
    monitor = 'val_loss',
    save_best_only = True,
    verbose = 1
)

print("🚀 Phase-1 training")
history = model.fit(train_ds, validation_data = val_ds, epochs = 30,
                    callbacks = [early_stop_phase1, reduce_lr_phase1, checkpoint_phase1, RichBatchProgressBar()], verbose = 0#, class_weight = class_weights
)

print("🔧 Phase-2 fine-tune")
base_model = model.layers[0]
for layer in base_model.layers[:-20]:
    layer.trainable = False
for layer in base_model.layers[-20:]:
    # 解凍但跳過 BatchNormalization
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True
    else:
        layer.trainable = False  # 保持 BN 推理模式
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-6),   
              loss = CategoricalCrossentropy(from_logits = False, label_smoothing = 0.1),    # label_smoothing = 0.1
              metrics = ['accuracy'])
history_fine = model.fit(
    train_ds, validation_data = val_ds, epochs = 20,
    callbacks = [early_stop_phase2, reduce_lr_phase2, checkpoint_phase2, RichBatchProgressBar()], verbose = 0#, class_weight = class_weights
)

for k in history.history:
    history.history[k].extend(history_fine.history[k])

# 重新載入 val_accuracy 最佳的模型
best_model_path = os.path.join(model_dir, f'best_val_acc_phase2_run{count}.h5')
if os.path.exists(best_model_path):
    print("🔁 Reload best Phase-2 model before final evaluation")
    model = load_model(best_model_path)

test_loss, test_acc = model.evaluate(test_ds, verbose = 1)
print(f"🧪 Test Acc: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

# 混淆矩陣
y_true = []
y_pred = []
for images, labels in test_ds:
    preds = model.predict(images, verbose = 0)
    y_true.extend(np.argmax(labels.numpy(), axis = 1))
    y_pred.extend(np.argmax(preds, axis = 1))

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = class_names)
fig, ax = plt.subplots(figsize = (8, 8))
disp.plot(ax = ax, cmap = 'Blues', values_format = 'd', colorbar = True)
plt.title(f"Confusion Matrix - Run {count}")
plt.grid(False)
plt.tight_layout()
cm_path = os.path.join(model_dir, f'confusion_matrix_run{count}.png')
plt.savefig(cm_path)
plt.show()
plt.close()
print(f"Confusion matrix saved to: {cm_path}")
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))
model.save(h5_path)
backup_model_path = os.path.join(model_dir, f'best_model_run{count}.h5')
model.save(backup_model_path)

plt.figure(figsize = (10, 6))
plt.plot(history.history['accuracy'], label = 'Train Acc')
plt.plot(history.history['val_accuracy'], label = 'Val Acc')
fine_tune_start_epoch = len(history.history['accuracy']) - len(history_fine.history['accuracy'])
plt.axvline(fine_tune_start_epoch - 1, color = 'red', linestyle = '--', label = 'Start Fine-tune')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.title(f'Training vs Validation Accuracy - Run {count}')
plt.legend()
plt.savefig(os.path.join(model_dir, f'train_plot_run{count}.png'))
plt.show()
plt.close()

plt.figure(figsize = (10, 6))
plt.plot(history.history['loss'], label = 'Train Loss')
plt.plot(history.history['val_loss'], label = 'Val Loss')
plt.axvline(fine_tune_start_epoch - 1, color = 'red', linestyle = '--', label = 'Start Fine-tune')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.title(f'Training vs Validation Loss - Run {count}')
plt.legend()
plt.savefig(os.path.join(model_dir, f'train_loss_run{count}.png'))
plt.show()
plt.close()

log_text  = f"Run {count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
log_text += f"Train Acc: {history.history['accuracy'][-1]:.4f} | Val Acc: {history.history['val_accuracy'][-1]:.4f}\n"
log_text += f"Test  Acc: {test_acc:.4f} | Test Loss: {test_loss:.4f}\n"
log_text += f"Train Loss: {history.history['loss'][-1]:.4f} | Val Loss: {history.history['val_loss'][-1]:.4f}\n"
log_text += f"Model Saved: {h5_path} and {backup_model_path}\n"
log_text += "-------------------------\n"

with open(os.path.join(model_dir, 'log.txt'), 'a', encoding = 'utf-8') as f:
    f.write(log_text)

# history_path = os.path.join(model_dir, f'history_run{count}.json')
# with open(history_path, 'w', encoding = 'utf-8') as f:
#     json.dump(history.history, f, indent = 2, ensure_ascii = False)
# print(f"Training history saved to: {history_path}")