import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras import callbacks
import json
import os

pd.set_option('future.no_silent_downcasting', True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Hàm 1: Tiền xử lý dữ liệu
def preprocess_data(data):
    data = data.dropna(subset=['num'])
    data['num'] = data['num'].apply(lambda x: 0 if x == 0 else 1)

    features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalch', 
                'exang', 'oldpeak', 'slope', 'ca', 'thal']
    target = 'num'

    data['fbs'] = data['fbs'].fillna(data['fbs'].mode()[0])
    data['exang'] = data['exang'].fillna(data['exang'].mode()[0])
    data['chol'] = data['chol'].replace(0, data['chol'][data['chol'] > 0].median())
    data['trestbps'] = data['trestbps'].replace(0, data['trestbps'].median())
    data['trestbps'] = data['trestbps'].fillna(data['trestbps'].median())
    data['chol'] = data['chol'].fillna(data['chol'].median())
    data['thalch'] = data['thalch'].fillna(data['thalch'].median())
    data['oldpeak'] = data['oldpeak'].fillna(data['oldpeak'].median())
    data['ca'] = data['ca'].fillna(data['ca'].median())
    data['thal'] = data['thal'].fillna(data['thal'].mode()[0])
    data['slope'] = data['slope'].fillna(data['slope'].mode()[0])

    binary_cols = ['sex', 'fbs', 'exang']
    encoders = {}
    for col in binary_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        encoders[col] = le

    data = pd.get_dummies(data, columns=['cp', 'restecg', 'slope', 'thal'], drop_first=False)
    features = [col for col in data.columns if col != target and col != 'id' and col != 'dataset']

    X = data[features]
    y = data[target]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"After preprocessing: {len(data)} samples, X shape: {X.shape}")
    print("Features:", features)
    return X_train, X_test, y_train, y_test, scaler, encoders

# Hàm 2: Xây dựng mô hình
def build_model(input_dim):
    model = Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Hàm 3: Huấn luyện mô hình
def train_model(model, X_train, y_train, class_weight=None, callbacks=None):
    history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_split=0.2, 
                        class_weight=class_weight, callbacks=callbacks, shuffle=True, verbose=1)
    return history

# Hàm 4: Đánh giá mô hình
def evaluate_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Độ chính xác trên tập kiểm tra: {accuracy * 100:.2f}%")
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    print("\nBáo cáo phân loại:\n", classification_report(y_test, y_pred))

def main():
    np.random.seed(42)
    tf.random.set_seed(42)

    data = pd.read_csv('heart_disease_uci.csv')
    X_train, X_test, y_train, y_test, scaler, encoders = preprocess_data(data)
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    
    model = build_model(input_dim=X_train.shape[1])
    print("Cấu trúc mô hình:\n")
    model.summary()

    n_class0 = sum(y_train == 0)
    n_class1 = sum(y_train == 1)
    n_samples = len(y_train)
    class_weight = {0: n_samples / (2 * n_class0), 1: n_samples / (2 * n_class1)}

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, min_delta=0.001)

    print("\nHuấn luyện mô hình...")
    train_model(model, X_train, y_train, class_weight=class_weight, callbacks=[early_stopping])
    
    print("\nĐánh giá mô hình...")
    evaluate_model(model, X_test, y_test)

    model.save('heart_disease_model.keras')
    print("Mô hình đã được lưu vào 'heart_disease_model.keras'")

    mean_list = list(scaler.mean_)
    std_list = list(scaler.scale_)
    with open('scaler_params.json', 'w') as f:
        json.dump({'mean': mean_list, 'std': std_list}, f)
    print("Scaler parameters đã được lưu vào 'scaler_params.json'")

if __name__ == "__main__":
    main()