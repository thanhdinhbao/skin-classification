import os, random, shutil, math
from glob import glob
from collections import Counter, defaultdict
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# ===================== Config =====================
ROOT = "Skin cancer ISIC The International Skin Imaging Collaboration"               # thư mục gốc
TRAIN_DIR = os.path.join(ROOT, "Train")
VAL_DIR   = os.path.join(ROOT, "Val")     # sẽ tạo nếu chưa có
TEST_DIR  = os.path.join(ROOT, "Test")

VAL_RATIO = 0.15               # lấy 15% từ Train làm Val
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 123
WARMUP_EPOCHS = 10
FINETUNE_EPOCHS = 15
PATIENCE = 5                    # early stop
LABEL_SMOOTHING = 0.1

# ===================== Utils ======================
def ensure_dir(p):
    if not os.path.exists(p): os.makedirs(p)

def stratified_split_train_to_val(train_dir, val_dir, ratio=0.15, seed=123):
    """Nếu chưa có Val, copy ngẫu nhiên ratio ảnh mỗi lớp từ Train sang Val (giữ nguyên Train)."""
    if os.path.exists(val_dir) and any(os.scandir(val_dir)):
        print(f"[Split] Val đã tồn tại, bỏ qua tách: {val_dir}")
        return
    print(f"[Split] Tạo Val từ Train với tỷ lệ {ratio:.0%}")
    random.seed(seed)
    for cls in sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]):
        src = os.path.join(train_dir, cls)
        dst = os.path.join(val_dir, cls)
        ensure_dir(dst)
        files = [f for f in glob(os.path.join(src, "*")) if os.path.isfile(f)]
        k = max(1, int(len(files) * ratio))
        pick = random.sample(files, k) if len(files) > 0 else []
        for f in pick:
            shutil.copy2(f, os.path.join(dst, os.path.basename(f)))
        print(f"  - {cls:30s} total={len(files)}  -> val={k}")

def count_per_class(folder):
    counts = {}
    for cls in sorted([d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]):
        n = len([f for f in glob(os.path.join(folder, cls, "*")) if os.path.isfile(f)])
        counts[cls] = n
    return counts

def compute_class_weights(counts_dict):
    """Trả về dict {class_index: weight} theo inverse frequency."""
    classes = sorted(counts_dict.keys())
    counts = np.array([counts_dict[c] for c in classes], dtype=np.float32)
    total = counts.sum()
    weights = total / (len(classes) * counts + 1e-8)
    return {i: float(w) for i, w in enumerate(weights)}, classes

# ===================== Split ======================
ensure_dir(ROOT)
ensure_dir(TRAIN_DIR)
ensure_dir(TEST_DIR)
ensure_dir(VAL_DIR)
stratified_split_train_to_val(TRAIN_DIR, VAL_DIR, VAL_RATIO, SEED)

# ===================== TF Datasets =================
AUTOTUNE = tf.data.AUTOTUNE

def make_ds(dir_path, subset=None, shuffle=True):
    return tf.keras.utils.image_dataset_from_directory(
        dir_path,
        labels="inferred",
        label_mode="categorical",
        class_names=None,              # lấy theo tên thư mục
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED
    )

train_ds = make_ds(TRAIN_DIR, shuffle=True)
val_ds   = make_ds(VAL_DIR,   shuffle=False)
test_ds  = make_ds(TEST_DIR,  shuffle=False)

# Lấy class_names theo train_ds
class_names = train_ds.class_names
num_classes = len(class_names)
print("Classes:", class_names)

# Pipeline tối ưu I/O
def prep(ds, training=False):
    if training:
        ds = ds.shuffle(1000, seed=SEED, reshuffle_each_iteration=True)
    return ds.prefetch(AUTOTUNE)

# Data augmentation layer (label-preserving)
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
])

# Chuẩn hoá theo VGG19
def preprocess(x, y):
    return preprocess_input(x), y

train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
train_ds = train_ds.map(preprocess, num_parallel_calls=AUTOTUNE)
val_ds   = val_ds.map(preprocess,   num_parallel_calls=AUTOTUNE)
test_ds  = test_ds.map(preprocess,  num_parallel_calls=AUTOTUNE)

train_ds = prep(train_ds, training=True)
val_ds   = prep(val_ds)
test_ds  = prep(test_ds)

# ===================== Class Weights ===============
train_counts = count_per_class(TRAIN_DIR)
class_weights, classes_sorted = compute_class_weights(train_counts)
print("Train counts:", train_counts)
print("Class weights (idx->w):", class_weights)

# ===================== Model =======================
base_model = VGG19(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
base_model.trainable = False

inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = inputs
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)
model = models.Model(inputs, outputs)

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
    metrics=["accuracy"]
)

ckpt_path = "vgg19_isic_best_warmup.h5"
cbs = [
    tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True, verbose=1),
]

print("\n======= Phase 1: Warm-up (freeze backbone) =======")
hist1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=WARMUP_EPOCHS,
    class_weight=class_weights,
    callbacks=cbs,
    verbose=1
)

# ===================== Fine-tune ====================
# Unfreeze một phần VGG19 (các block cuối)
for layer in base_model.layers:
    layer.trainable = False
# mở khóa từ block5_conv1 trở đi (điều chỉnh theo nhu cầu)
set_trainable = False
for layer in base_model.layers[::-1]:
    if "block5" in layer.name:
        set_trainable = True
    if set_trainable and not isinstance(layer, layers.BatchNormalization):
        layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=1e-4),  # nhỏ hơn khi fine-tune
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
    metrics=["accuracy"]
)

ckpt_path_ft = "vgg19_isic_best_finetune.h5"
cbs_ft = [
    tf.keras.callbacks.ModelCheckpoint(ckpt_path_ft, monitor="val_accuracy", save_best_only=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True, verbose=1),
]

print("\n======= Phase 2: Fine-tune (unfreeze top blocks) =======")
hist2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINETUNE_EPOCHS,
    class_weight=class_weights,
    callbacks=cbs_ft,
    verbose=1
)

# ====== Gộp lịch sử 2 phase ======
def combine_hist(h1, h2):
    hist = {}
    for k in h1.history.keys():
        hist[k] = h1.history[k] + h2.history[k]
    return hist

history = combine_hist(hist1, hist2)

# ====== Vẽ Accuracy ======
plt.figure(figsize=(8, 5))
plt.plot(history['accuracy'], label='Train Acc')
plt.plot(history['val_accuracy'], label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training & Validation Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# ====== Vẽ Loss ======
plt.figure(figsize=(8, 5))
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

# ===================== Evaluate =====================
print("\nEvaluating on TEST set...")
test_metrics = model.evaluate(test_ds, verbose=1)
print("Test metrics:", dict(zip(model.metrics_names, test_metrics)))

model.save("vgg19_skin_lesion_model_finetuned.h5")
print("Saved model -> vgg19_skin_lesion_model_finetuned.h5")