import os
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import MobileNetV2, mobilenet_v2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from datetime import datetime
import json
import sys
sys.stdout.flush()

# 模型與輸出資料夾
model_dir = r'C:\E521C\saved_models_v2_new'
os.makedirs(model_dir, exist_ok = True)
count_path = os.path.join(model_dir, 'train_count.txt')

# 訓練次數紀錄
if os.path.exists(count_path):
    with open(count_path, 'r') as f:
        count = int(f.read().strip()) + 1
else:
    count = 1
with open(count_path, 'w') as f:
    f.write(str(count))
print(f"🧠 Training No.{count}", flush = True)

# 手動分好之資料夾
train_dir = r'C:\E521C\DataImage\trash_dataset_v2_new_split\train'
val_dir   = r'C:\E521C\DataImage\trash_dataset_v2_new_split\val'
test_dir  = r'C:\E521C\DataImage\trash_dataset_v2_new_split\test'

# 讀取資料集
train_ds = tf.keras.utils.image_dataset_from_directory(train_dir, image_size = (128, 128), batch_size = 16, label_mode = 'int')
val_ds   = tf.keras.utils.image_dataset_from_directory(val_dir, image_size = (128, 128), batch_size = 16, label_mode = 'int')
test_ds  = tf.keras.utils.image_dataset_from_directory(test_dir, image_size = (128, 128), batch_size = 16, label_mode = 'int')

# 類別數與名稱
num_classes = len(train_ds.class_names)
class_names = train_ds.class_names
print(f"Number of classes: {num_classes}")

# 前處理與增強
augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.1),
    layers.RandomTranslation(0.1, 0.1)
])

def preprocess(x, y):
    return mobilenet_v2.preprocess_input(x), y

train_ds = train_ds.map(lambda x, y: (augmentation(x), y)).map(preprocess)
val_ds   = val_ds.map(preprocess)
test_ds  = test_ds.map(preprocess)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size = AUTOTUNE)
val_ds   = val_ds.prefetch(buffer_size = AUTOTUNE)
test_ds  = test_ds.prefetch(buffer_size = AUTOTUNE)

# 模型路徑
h5_path    = os.path.join(model_dir, 'trash_model_advanced.h5')
keras_path = os.path.join(model_dir, 'trash_model_advanced.keras')

def build_model():
    base_model = MobileNetV2(input_shape = (128, 128, 3), include_top = False, weights = 'imagenet')
    base_model.trainable = True
    for layer in base_model.layers[:120]:
        layer.trainable = False
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation = 'relu', kernel_regularizer = regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(128, activation = 'relu', kernel_regularizer = regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation = 'softmax')
    ])
    return model

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
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False),
    metrics = ['accuracy']
)

early_stop = EarlyStopping(monitor = 'val_accuracy', patience = 5, restore_best_weights = True)
reduce_lr  = ReduceLROnPlateau(monitor = 'val_accuracy', factor = 0.5, patience = 3, min_lr = 1e-6, verbose = 1)
checkpoint = ModelCheckpoint(filepath = os.path.join(model_dir, 'best_val_acc.h5'),
                             monitor = 'val_accuracy', save_best_only = True, verbose = 1)

print("🚀 Phase-1 training")
history = model.fit(train_ds, validation_data = val_ds, epochs = 50,
                    callbacks = [early_stop, reduce_lr, checkpoint], verbose = 2)

print("🔧 Phase-2 fine-tune (unfreeze all)")
model.layers[0].trainable = True
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-5),
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False),
              metrics = ['accuracy'])
history_fine = model.fit(train_ds, validation_data = val_ds, epochs = 10,
                         callbacks = [early_stop, reduce_lr, checkpoint], verbose = 2)

for k in history.history:
    history.history[k].extend(history_fine.history[k])

test_loss, test_acc = model.evaluate(test_ds, batch_size = 16, verbose = 0)
print(f"🧪 Test Acc: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

# 混淆矩陣
y_true = []
y_pred = []
for images, labels in test_ds:
    preds = model.predict(images, verbose = 0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis = 1))

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = class_names)
fig, ax = plt.subplots(figsize = (8, 8))
disp.plot(ax = ax, cmap = 'Blues', values_format = 'd')
plt.title(f"Confusion Matrix - Run {count}")
plt.grid(False)
plt.tight_layout()
cm_path = os.path.join(model_dir, f'confusion_matrix_run{count}.png')
plt.savefig(cm_path)
plt.show()
plt.close()
print(f"📊 Confusion matrix saved to: {cm_path}")

model.save(h5_path)
backup_model_path = os.path.join(model_dir, f'best_model_run{count}.h5')
model.save(backup_model_path)

plt.plot(history.history['accuracy'], label = 'Train Acc')
plt.plot(history.history['val_accuracy'], label = 'Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.title(f'Training vs Validation Accuracy - Run {count}')
plt.legend()
plt.savefig(os.path.join(model_dir, f'train_plot_run{count}.png'))
plt.show()
plt.close()

plt.plot(history.history['loss'], label = 'Train Loss')
plt.plot(history.history['val_loss'], label = 'Val Loss')
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

history_path = os.path.join(model_dir, f'history_run{count}.json')
with open(history_path, 'w', encoding = 'utf-8') as f:
    json.dump(history.history, f, indent = 2, ensure_ascii = False)
print(f"📁 Training history saved to: {history_path}")