import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
import json
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import MobileNetV2, mobilenet_v2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from datetime import datetime
from rich.console import Console
from rich.traceback import install

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all, 1 = info, 2 = warning, 3 = error only
tf.get_logger().setLevel('ERROR')
console = Console() 
sys.stdout.flush()
install()

# ===== Random Seed =====
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)

# ===== Model Saved =====
model_dir = r"C:\E521C\Model\model_smoothing"     # model save path
os.makedirs(model_dir, exist_ok = True)
count_path = os.path.join(model_dir, 'train_count.txt')

# ===== Train Count =====
if os.path.exists(count_path):
    with open(count_path, 'r') as f:
        count = int(f.read().strip()) + 1
else:
    count = 1
with open(count_path, 'w') as f:
    f.write(str(count))

console.print(f"[cyan]Training No.{count}[/]")

# ===== Size Setting =====
BS = 32         # batch_size
size = 192      # image_size
IS = (size, size)

# ===== Dataset Path Setting =====
train_dir = r"C:\E521C\Dataset\trash_dataset_v2_new_split\train"  
val_dir   = r"C:\E521C\Dataset\trash_dataset_v2_new_split\val"
test_dir  = r"C:\E521C\Dataset\trash_dataset_v2_new_split\test"

def load_dataset(path):
    return tf.keras.utils.image_dataset_from_directory(
        path,
        image_size = IS,
        batch_size = BS,
        label_mode = 'categorical'
    )
train_ds = load_dataset(train_dir)
val_ds   = load_dataset(val_dir)
test_ds  = load_dataset(test_dir)

# ===== Print Data-count Setting =====
def count_images(directory):
    return sum([len(files) for _, _, files in os.walk(directory) if any(f.lower().endswith(('jpg','png','jpeg')) for f in files)])

num_train = count_images(train_dir)
num_val   = count_images(val_dir)
num_test  = count_images(test_dir)
num_classes = len(train_ds.class_names)

console.print(f"[red]Train: {num_train} files | {num_classes} classes[/]")
console.print(f"[yellow]Val: {num_val} files | {num_classes} classes[/]")
console.print(f"[green]Test: {num_test} files | {num_classes} classes[/]")

num_classes = len(train_ds.class_names)
class_names = train_ds.class_names
print(f"Number of classes: {num_classes}")
print(class_names)

# ===== Class Weight =====
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
    console.print("[red] Warning! class_weight set at 1[/]")
    for i in range(num_classes):
        if i not in class_weights:
            class_weights[i] = 1.0

print("Computed class weights:", class_weights)

# ===== Data Augmentation and Preprocessing Pipeline =====
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
def preprocess_with_aug(x, y):
    x = augmentation(x)                        
    x = mobilenet_v2.preprocess_input(x)       
    return x, y

# ===== Mixup =====
mixup_alpha = 0.15
def mixup(ds, alpha = mixup_alpha):
    def mix(x, y):
        beta = tf.random.uniform([], 0, alpha)
        index = tf.random.shuffle(tf.range(tf.shape(x)[0]))
        x_mix = beta * x + (1 - beta) * tf.gather(x, index)
        y_mix = beta * y + (1 - beta) * tf.gather(y, index)
        return x_mix, y_mix
    return ds.map(mix, num_parallel_calls = AUTOTUNE)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(lambda x, y: preprocess_with_aug(x, y), num_parallel_calls = AUTOTUNE)
train_ds = mixup(train_ds).prefetch(buffer_size = AUTOTUNE)
val_ds = val_ds.map(lambda x, y: preprocess(x, y), num_parallel_calls = AUTOTUNE)
test_ds = test_ds.map(preprocess, num_parallel_calls = AUTOTUNE).prefetch(buffer_size = AUTOTUNE)

h5_path    = os.path.join(model_dir, 'trash_model_advanced.h5')
keras_path = os.path.join(model_dir, 'trash_model_advanced.keras')

# ===== Dense Setting =====
def build_model():
    base_model = MobileNetV2(input_shape = (192, 192, 3), include_top = False, weights = 'imagenet')    # image size
    for layer in base_model.layers:
        layer.trainable = False
    model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation = 'relu', kernel_regularizer = regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(128, activation = 'relu', kernel_regularizer = regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax', kernel_regularizer = regularizers.l2(1e-4))
])
    return model

# ===== Progress Bar Setting =====
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

# ===== Label Smoothing =====
def custom_loss(y_true, y_pred):
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred, label_smoothing = 0.01) # label smoothing

# ===== Cosine Decay LR Schedule =====
total_epochs_phase1 = 30
steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
lr_schedule_phase1 = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate = 1e-3,
    decay_steps = total_epochs_phase1 * steps_per_epoch,
    alpha = 1e-5
)
total_epochs_phase2 = 30
lr_schedule_phase2 = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate = 1e-5,    
    decay_steps = total_epochs_phase2 * steps_per_epoch,
    alpha = 1e-6
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

# ===== Parameter Setting =====
early_stop_phase1 = EarlyStopping(monitor = 'val_accuracy', patience = 5, restore_best_weights = True, verbose = 1)
early_stop_phase2 = EarlyStopping(monitor = 'val_accuracy', patience = 5, restore_best_weights = True, verbose = 1)
# reduce_lr_phase1 = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 3, min_lr = 1e-6, verbose = 1)
# reduce_lr_phase2 = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.3, patience = 3, min_lr = 1e-7, verbose = 1)

# ===== Checkpoint =====
checkpoint_phase1 = ModelCheckpoint(
    filepath = os.path.join(model_dir, f'best_val_acc_phase1_run{count}.h5'),
    monitor = 'val_loss',   
    save_best_only = True,
    verbose = 1
)
checkpoint_phase2 = ModelCheckpoint(
    filepath = os.path.join(model_dir, f'best_val_acc_phase2_run{count}.h5'),
    monitor = 'val_loss',
    save_best_only = True,
    verbose = 1
)

# ===== Phase-1 =====
console.print("[magenta]Phase-1 training[/]")
history = model.fit(
    train_ds, validation_data = val_ds, epochs = 30,    # epoch
    callbacks = [early_stop_phase1, checkpoint_phase1, RichBatchProgressBar()], verbose = 0, class_weight = class_weights
)

# ===== Phase-2 =====
console.print("[magenta]Phase-2 fine-tune[/]")
base_model = model.layers[0]
for layer in base_model.layers[:-20]:
    layer.trainable = False
set_trainable = False
for layer in base_model.layers:
    if 'block_5' in layer.name:     # unfreeze layer
        set_trainable = True
    layer.trainable = set_trainable and not isinstance(layer, tf.keras.layers.BatchNormalization)

class LRSchedulerLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs = None):
        lr = self.model.optimizer._decayed_lr(tf.float32).numpy()
        print(f"[LR] Epoch {epoch + 1}: {lr:.8f}")

model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule_phase2),
    loss = custom_loss,
    metrics = ['accuracy']
)
history_fine = model.fit(
    train_ds, validation_data = val_ds, epochs = 30,    # epoch
    callbacks = [early_stop_phase2, checkpoint_phase2, RichBatchProgressBar(), LRSchedulerLogger()], verbose = 0, class_weight = class_weights
)
for k in history.history:
    history.history[k].extend(history_fine.history[k])

# ===== Reload the Best Model =====
best_model_path = os.path.join(model_dir, f'best_val_acc_phase2_run{count}.h5')
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
model.save(h5_path, save_format = 'h5')
model.save(keras_path, save_format = 'keras')
backup_model_path = os.path.join(model_dir, f'best_model_run{count}.h5')
model.save(backup_model_path, save_format = 'h5')

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

# ===== Log Text =====
log_text  = f"Run {count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
log_text += f"Train Acc: {history.history['accuracy'][-1]:.4f} | Val Acc: {history.history['val_accuracy'][-1]:.4f}\n"
log_text += f"Test  Acc: {test_acc:.4f} | Test Loss: {test_loss:.4f}\n"
log_text += f"Train Loss: {history.history['loss'][-1]:.4f} | Val Loss: {history.history['val_loss'][-1]:.4f}\n"
log_text += f"Model Saved: {h5_path} and {backup_model_path}\n"
log_text += "-------------------------\n"
with open(os.path.join(model_dir, 'log.txt'), 'a', encoding = 'utf-8') as f:
    f.write(log_text)

# ===== .json =====
def convert_history(history_dict):
    return {k: [float(vv) for vv in v] for k, v in history_dict.items()}
history_path = os.path.join(model_dir, f'history_run{count}.json')
with open(history_path, 'w', encoding = 'utf-8') as f:
    json.dump(history.history, f, indent = 2, ensure_ascii = False, default = lambda o: float(o) if isinstance(o, (np.float32, np.float64)) else o)

# ===== Saved class_names to .json =====
class_names_path = os.path.join(model_dir, "class_names.json")
with open(class_names_path, "w", encoding="utf-8") as f:
    json.dump(class_names, f)
console.print(f"[green]Class names saved to: {class_names_path}[/]")

# ===== TF-lite =====
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()
# with open(os.path.join(model_dir, f"model_run{count}.tflite"), 'wb') as f:
#     f.write(tflite_model)