import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
# ===== CONFIG =====
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 5

base_path = os.getcwd()
train_dir = os.path.join(base_path, "dataset/train")
test_dir = os.path.join(base_path, "dataset/test")  # 👉 THÊM TEST

# ===== DATA =====
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    color_mode='rgb'
)

val_data = datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    color_mode='rgb'
)

# ===== TEST DATA =====
test_dir = os.path.join(base_path, "dataset/test")

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False   # 🔥 QUAN TRỌNG
)

# ===== MODEL =====
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
output = layers.Dense(train_data.num_classes, activation='softmax')(x)

model = models.Model(inputs=base_model.input, outputs=output)

# ===== COMPILE =====
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ===== CALLBACK =====
os.makedirs("model", exist_ok=True)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "model/best_model.keras",
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# ===== TRAIN =====
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[checkpoint]
)

print("✅ DONE TRAIN")

# ===== VẼ BIỂU ĐỒ =====

# Accuracy
plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("model/accuracy.png")
plt.show()

# Loss
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig("model/loss.png")
plt.show()

# ===== CONFUSION MATRIX =====

print("🔍 Evaluating on test set...")

# Dự đoán
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

# Nhãn thật
y_true = test_generator.classes

# Tạo ma trận
cm = confusion_matrix(y_true, y_pred_classes)

# Hiển thị
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=list(test_generator.class_indices.keys())
)

plt.figure(figsize=(8,6))
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")

# Lưu file (để chèn báo cáo)
plt.savefig("model/confusion_matrix.png")
plt.tight_layout()
plt.draw()
plt.show()
# ===== SAVE MODEL =====
model.save("model/fashion_model.h5")

print("✅ DONE ALL")