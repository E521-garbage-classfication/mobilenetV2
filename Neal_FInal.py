import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
import json
import math
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import MobileNetV2, mobilenet_v2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from datetime import datetime
from rich.console import Console
from rich.traceback import install

# ===== Quiet TensorFlow logs =====
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
console = Console()
sys.stdout.flush()
install()

# ===== Reproducibility =====
seed = 77
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)

# ===== Paths / Run counter =====
model_dir = r"C:\E521C\Model\model"  # Model save path
os.makedirs(model_dir, exist_ok = True)
count_path = os.path.join(model_dir, 'train_count.txt')

if os.path.exists(count_path):
    with open(count_path, 'r') as f:
        count = int(f.read().strip()) + 1
else:
    count = 1
with open(count_path, 'w') as f:
    f.write(str(count))
console.print(f"[cyan]Training No.{count}[/]")

# ===== Dataset Path Setting =====
train_dir = r"C:\E521C\Dataset\Classified\train"
val_dir   = r"C:\E521C\Dataset\Classified\val"
test_dir  = r"C:\E521C\Dataset\Classified\test"

# ===== Data config =====
BS = 16               # Batch size
size = 192            # Image size
IS = (size, size)

def load_dataset(path, shuffle, seed = None):
    return tf.keras.utils.image_dataset_from_directory(
        path,
        image_size = IS,
        batch_size = BS,
        label_mode = 'categorical',
        shuffle = shuffle,
        seed = seed
    )

train_ds = load_dataset(train_dir, shuffle = True, seed = seed)
val_ds   = load_dataset(val_dir, shuffle = False)
test_ds  = load_dataset(test_dir, shuffle = False)

# ===== Print Data-count Setting =====
def count_images(directory):
    exts = ('.jpg', '.jpeg', '.png')
    total = 0
    for _, _, files in os.walk(directory):
        total += sum(1 for f in files if f.lower().endswith(exts))
    return total

num_train = count_images(train_dir)
num_val   = count_images(val_dir)
num_test  = count_images(test_dir)

num_classes = len(train_ds.class_names)
class_names = train_ds.class_names
console.print(f"[red]Train: {num_train} files | {num_classes} classes[/]")
console.print(f"[yellow]Val: {num_val} files | {num_classes} classes[/]")
console.print(f"[green]Test: {num_test} files | {num_classes} classes[/]")
print(f"Number of classes: {num_classes}")
print(class_names)

# ===== Class weights (from original labels, before augmentation) =====
y_train_all = []
for _, labels in train_ds.unbatch():
    y_train_all.append(np.argmax(labels.numpy()))
y_train_all = np.array(y_train_all)

class_weights_array = compute_class_weight(
    class_weight = 'balanced',
    classes = np.unique(y_train_all),
    y = y_train_all
)
class_weights = dict(enumerate(class_weights_array))
if len(np.unique(y_train_all)) < num_classes:
    console.print("[red] Warning! some classes missing in y_train; setting their weight to 1.0[/]")
    for i in range(num_classes):
        if i not in class_weights:
            class_weights[i] = 1.0
print("Computed class weights:", class_weights)

# Clip class weights to avoid over-amplifying any single class
for k, v in class_weights.items():
    class_weights[k] = float(np.clip(v, 0.5, 2.0))
print("Clipped class weights:", class_weights)


# ===== Data augmentation & preprocessing =====
# Note: apply photometric/spatial augs before preprocess_input.
# GaussianNoise should be applied AFTER normalization; see preprocess_with_aug.
augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomBrightness(0.2),
    layers.RandomContrast(0.2),
    layers.RandomTranslation(0.1, 0.1)
])

# ===== Gaussian noise on normalized [-1, 1] images =====
noise_layer = layers.GaussianNoise(0.03)

def preprocess(x, y):
    x = mobilenet_v2.preprocess_input(x)
    return x, y

def preprocess_with_aug(x, y):
    x = augmentation(x)
    x = mobilenet_v2.preprocess_input(x)
    x = noise_layer(x, training = True)
    return x, y

light_aug = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomBrightness(0.08),
    layers.RandomContrast(0.08),
    layers.GaussianNoise(0.02),
])
def preprocess_with_light_aug(x, y):
    x = light_aug(x, training=True)
    x = mobilenet_v2.preprocess_input(x)
    return x, y

# ===== MixUp =====
def sample_beta(alpha):
    x = tf.random.gamma(shape = [], alpha = alpha, beta = 1.0)
    y = tf.random.gamma(shape = [], alpha = alpha, beta = 1.0)
    return x / (x + y)

mixup_alpha = 0.1
def mixup(ds, alpha = mixup_alpha):
    def mix(x, y):
        lam = sample_beta(alpha)
        index = tf.random.shuffle(tf.range(tf.shape(x)[0]))
        x_mix = lam * x + (1 - lam) * tf.gather(x, index)
        y_mix = lam * y + (1 - lam) * tf.gather(y, index)
        return x_mix, y_mix
    return ds.map(mix, num_parallel_calls = AUTOTUNE)

mixup_alpha_ft = 0.05

def mixup_ft(ds, alpha = mixup_alpha_ft):
    def mix(x, y):
        lam = sample_beta(alpha)
        idx = tf.random.shuffle(tf.range(tf.shape(x)[0]))
        return lam * x + (1 - lam) * tf.gather(x, idx), lam * y + (1 - lam) * tf.gather(y, idx)
    return ds.map(mix, num_parallel_calls = AUTOTUNE)

AUTOTUNE = tf.data.AUTOTUNE

# Phase-1
train_ds_phase1 = train_ds.map(lambda x, y: preprocess_with_aug(x, y), num_parallel_calls = AUTOTUNE)
train_ds_phase1 = mixup(train_ds_phase1).prefetch(AUTOTUNE)

# Phase-2
train_ds_phase2 = train_ds.map(preprocess_with_light_aug, num_parallel_calls = AUTOTUNE)
train_ds_phase2 = mixup_ft(train_ds_phase2).prefetch(AUTOTUNE)

# Validation / Test: no augmentation
val_ds  = val_ds.map(lambda x, y: preprocess(x, y), num_parallel_calls = AUTOTUNE).cache().prefetch(AUTOTUNE)
test_ds = test_ds.map(preprocess, num_parallel_calls = AUTOTUNE).cache().prefetch(AUTOTUNE)

h5_path    = os.path.join(model_dir, 'trash_model_advanced.h5')
keras_path = os.path.join(model_dir, 'trash_model_advanced.keras')

# ===== Model definition =====
def build_model():
    base_model = MobileNetV2(input_shape = (192, 192, 3), include_top = False, weights = 'imagenet')
    for layer in base_model.layers:
        layer.trainable = False
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation = 'relu', kernel_regularizer = regularizers.l2(2e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(64, activation = 'relu', kernel_regularizer = regularizers.l2(2e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation = 'softmax', kernel_regularizer = regularizers.l2(2e-4))
    ])
    return model

# ===== Helper: current LR (schedule or constant) =====
def get_current_lr(optimizer):
    """
    Return the effective learning rate at the current step.
    Supports:
    - constant/variable learning_rate (float or tf.Variable)
    - LearningRateSchedule (e.g., CosineDecay, CosineDecayRestarts)
    """
    lr_obj = optimizer.learning_rate
    if isinstance(lr_obj, LearningRateSchedule):
        value = lr_obj(optimizer.iterations)
        return float(tf.keras.backend.get_value(value))
    else:
        return float(tf.keras.backend.get_value(lr_obj))

# ===== Rich progress bar =====
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
        if hasattr(self, 'progress'):
            try:
                self.progress.stop()
            except Exception:
                pass

# ===== Loss (label smoothing) =====
def custom_loss(y_true, y_pred):
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred, label_smoothing = 0.03)

# ===== Schedules =====
total_epochs_phase1 = 40
card = tf.data.experimental.cardinality(train_ds).numpy()
steps_per_epoch = card if card > 0 else math.ceil(num_train / BS)
print("steps_per_epoch:", steps_per_epoch)

lr_schedule_phase1 = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate = 1e-3,
    decay_steps = total_epochs_phase1 * steps_per_epoch,
    alpha = 1e-5
)

total_epochs_phase2 = 30
lr_schedule_phase2 = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate = 5e-6,
    decay_steps = total_epochs_phase2 * steps_per_epoch,
    alpha = 1e-7
)

# ===== Model =====
try:
    if os.path.exists(h5_path):
        console.print("[cyan]Load existing .h5 model[/]")
        model = load_model(h5_path, custom_objects = {'custom_loss': custom_loss})
    elif os.path.exists(keras_path):
        console.print("[cyan]Convert .keras → .h5[/]")
        model = load_model(keras_path, custom_objects = {'custom_loss': custom_loss})
        model.save(h5_path)
    else:
        raise FileNotFoundError
except Exception as e:
    console.print("[cyan]Build new model:[/]", e)
    model = build_model()

model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule_phase1),
    loss = custom_loss,
    metrics = ['accuracy']
)

# ===== Early-stop =====
early_stop_phase1 = EarlyStopping(monitor = 'val_loss', patience = 7, min_delta = 1e-3, restore_best_weights = True, verbose = 1)
early_stop_phase2 = EarlyStopping(monitor = 'val_loss', patience = 5, min_delta = 1e-3, restore_best_weights = True, verbose = 1)

# ===== Checkpoint =====
checkpoint_phase1 = ModelCheckpoint(
    filepath = os.path.join(model_dir, f'best_val_loss_phase1_run{count}.h5'),
    monitor = 'val_loss',
    save_best_only = True,
    verbose = 1
)
checkpoint_phase2 = ModelCheckpoint(
    filepath = os.path.join(model_dir, f'best_val_loss_phase2_run{count}.h5'),
    monitor = 'val_loss',
    save_best_only = True,
    verbose = 1
)

# ===== Phase 1 =====
console.print("[magenta]Phase-1 training[/]")
history = model.fit(
    train_ds_phase1, validation_data=val_ds, epochs = 40,
    callbacks = [early_stop_phase1, checkpoint_phase1, RichBatchProgressBar()],
    verbose = 0, class_weight = class_weights
)


# ===== Phase 2 (fine-tuning) =====
console.print("[magenta]Phase-2 fine-tune[/]")
base_model = model.layers[0]
unfreeze = False
for layer in base_model.layers:
    if 'block_13' in layer.name:    # unfreeze
        unfreeze = True
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False
    else:
        layer.trainable = unfreeze

class LRSchedulerLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs = None):
        lr = get_current_lr(self.model.optimizer)
        print(f"[LR] Epoch {epoch + 1}: {lr:.8f}")

model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule_phase2),
    loss = custom_loss,
    metrics = ['accuracy']
)

history_fine = model.fit(
    train_ds_phase2, validation_data = val_ds, epochs = 30,
    callbacks = [early_stop_phase2, checkpoint_phase2, RichBatchProgressBar(), LRSchedulerLogger()],
    verbose = 0# , class_weight = class_weights  # 若仍偏向 metal，可做一輪 A/B 將此參數移除
)

for k in history.history:
    history.history[k].extend(history_fine.history[k])

# ===== Reload best Phase-2 model for evaluation/export =====
best_model_path = os.path.join(model_dir, f'best_val_loss_phase2_run{count}.h5')
if os.path.exists(best_model_path):
    console.print("[magenta]Reload best Phase-2 model before final evaluation[/]")
    model = load_model(best_model_path, custom_objects = {'custom_loss': custom_loss})

test_loss, test_acc = model.evaluate(test_ds, verbose = 1)
print(f"Test Acc: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

# ===== Confusion-Matrix =====
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
console.print("[cyan]Classification Report:[/]\n")
report_text = classification_report(y_true, y_pred, target_names = class_names)
print(report_text)
with open(os.path.join(model_dir, f'classification_report_run{count}.txt'), 'w', encoding = 'utf-8') as f:
    f.write(report_text)

model.save(h5_path)
model.save(keras_path)
backup_model_path = os.path.join(model_dir, f'best_model_run{count}.h5')
model.save(backup_model_path)

# ===== Accuracy Chart =====
plt.figure(figsize = (10, 6))
plt.plot(history.history['accuracy'], label = 'Train Acc')
plt.plot(history.history['val_accuracy'], label = 'Val Acc')
fine_tune_start_epoch = len(history.history['accuracy']) - len(history_fine.history['accuracy'])
plt.axvline(fine_tune_start_epoch - 1, color = 'red', linestyle = '--', label = 'Start Fine-tune')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.title(f'[Train/Val Acc] Run {count} ({datetime.now().strftime("%m/%d %H:%M")})')
plt.legend()
plt.savefig(os.path.join(model_dir, f'train_acc_run{count}.png'))
plt.show()
plt.close()

# ===== Loss Chart =====
plt.figure(figsize = (10, 6))
plt.plot(history.history['loss'], label = 'Train Loss')
plt.plot(history.history['val_loss'], label = 'Val Loss')
plt.axvline(fine_tune_start_epoch - 1, color = 'red', linestyle = '--', label = 'Start Fine-tune')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.title(f'Training vs Validation Loss - Run {count} ({datetime.now().strftime("%m/%d %H:%M")})')
plt.legend()
plt.savefig(os.path.join(model_dir, f'train_loss_run{count}.png'))
plt.show()
plt.close()

# ===== Log text =====
log_text  = f"Run {count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
log_text += f"Train Acc: {history.history['accuracy'][-1]:.4f} | Val Acc: {history.history['val_accuracy'][-1]:.4f}\n"
log_text += f"Test  Acc: {test_acc:.4f} | Test Loss: {test_loss:.4f}\n"
log_text += f"Train Loss: {history.history['loss'][-1]:.4f} | Val Loss: {history.history['val_loss'][-1]:.4f}\n"
log_text += f"Model Saved: {h5_path} and {backup_model_path}\n"
log_text += "-------------------------\n"
with open(os.path.join(model_dir, 'log.txt'), 'a', encoding = 'utf-8') as f:
    f.write(log_text)

# ===== Save training history & class names =====
# def convert_history(history_dict):
#     return {k: [float(vv) for vv in v] for k, v in history_dict.items()}
history_path = os.path.join(model_dir, f'history_run{count}.json')
with open(history_path, 'w', encoding = 'utf-8') as f:
    json.dump(history.history, f, indent = 2, ensure_ascii = False, default = lambda o: float(o) if isinstance(o, (np.float32, np.float64)) else o)

class_names_path = os.path.join(model_dir, "class_names.json")
with open(class_names_path, "w", encoding = "utf-8") as f:
    json.dump(class_names, f)
console.print(f"[green]Class names saved to: {class_names_path}[/]")

# ===== TFLite export =====
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open(os.path.join(model_dir, f"model_run{count}.tflite"), 'wb') as f:
    f.write(tflite_model)