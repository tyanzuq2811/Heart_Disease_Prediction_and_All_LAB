import tensorflow as tf
from tensorflow.keras.models import load_model

# Tải mô hình từ file .keras
model = load_model('heart_disease_model.keras')

# 1. Xem kiến trúc mô hình
print("Kiến trúc mô hình:")
model.summary()

# 2. Xem trọng số của từng lớp
print("\nTrọng số của từng lớp:")
for layer in model.layers:
    weights = layer.get_weights()
    if weights:  # Chỉ in nếu lớp có trọng số
        print(f"Layer: {layer.name}")
        for i, w in enumerate(weights):
            print(f"  Weight {i}: shape = {w.shape}")
            print(f"  Values:\n{w}\n")
    else:
        print(f"Layer: {layer.name} (không có trọng số)")

# 3. Xem cấu hình huấn luyện
print("\nCấu hình huấn luyện:")
print(f"Hàm mất mát: {model.loss}")
print(f"Bộ tối ưu: {model.optimizer}")
print(f"Chỉ số: {model.metrics_names}")