# main_pipeline.py
# Гибридная система + LSTM для прогноза RUL
import os
import sys
import time
import subprocess
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_absolute_error, confusion_matrix, precision_recall_curve

tf.random.set_seed(42)

def create_sequences(data, targets, seq_length=50):
    """Создание последовательностей для LSTM"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(targets[i + seq_length])
    return np.array(X), np.array(y)

def run_cpp_generator():
    print("="*60)
    print(" ЭТАП 1: Генерация синтетических данных (C++)")
    print("="*60)
    exe_name = "GEN.exe"
    if not os.path.exists(exe_name):
        print(f" ОШИБКА: {exe_name} не найден.")
        return False
    start = time.time()
    try:
        result = subprocess.run([exe_name], capture_output=True, text=True)
        print(f" Время генерации: {time.time()-start:.3f} сек.")
        if result.returncode == 0:
            print(" satellite_logs.csv создан")
            return True
        return False
    except Exception as e:
        print(f" {e}")
        return False

def run_ml_pipeline():
    print("\n" + "="*60)
    print(" ЭТАП 2: Обработка данных и Обучение")
    print("="*60)
    
    print(" Загрузка и Feature Engineering...")
    df = pd.read_csv("satellite_logs.csv")
    df = df.sort_values(['device_id', 'timestamp'])

    # признаки
    df['cno_ma5'] = df.groupby('device_id')['cno'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df['cno_anomaly'] = df['cno'] - df['cno_ma5']
    df['temp_stress'] = (df['temperature'] - 40.0).abs()
    df['drift_rate'] = df.groupby('device_id')['timing_drift'].diff().fillna(0)

    BASE_FEATURES = ['temperature', 'voltage', 'cno', 'pseudorange_res', 'timing_drift', 'hours_since_maint', 'solar_activity']
    NEW_FEATURES = ['cno_ma5', 'cno_anomaly', 'temp_stress', 'drift_rate']
    FEATURES = BASE_FEATURES + NEW_FEATURES
    print(f" Признаков: {len(FEATURES)} (добавлено: {NEW_FEATURES})")

    X = df[FEATURES].values
    y_rul = df['rul'].values
    y_risk = df['integrity_risk'].values

    # Масштабирование для LSTM
    scaler_lstm = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler_lstm.fit_transform(X)
    
    scaler_rul = MinMaxScaler(feature_range=(0, 1))
    y_rul_scaled = scaler_rul.fit_transform(y_rul.reshape(-1, 1)).flatten()

    # Создание последовательностей
    SEQ_LENGTH = 50
    print(f"\n Создание последовательностей (окно {SEQ_LENGTH} часов)...")
    X_lstm, y_lstm = create_sequences(X_scaled, y_rul_scaled, SEQ_LENGTH)
    print(f" Последовательностей создано: {X_lstm.shape[0]}")

    # Разделение для LSTM
    X_lstm_train, X_lstm_test, y_lstm_train, y_lstm_test = train_test_split(
        X_lstm, y_lstm, test_size=0.2, random_state=42, shuffle=False
    )

    # Для классификации (NN)
    X_train_cls, X_test_cls, y_train_risk, y_test_risk = train_test_split(
        X, y_risk, test_size=0.2, random_state=42
    )
    
    scaler_cls = StandardScaler()
    X_train_cls_sc = scaler_cls.fit_transform(X_train_cls)
    X_test_cls_sc = scaler_cls.transform(X_test_cls)

    # ================= RANDOM FOREST (Классификатор) =================
    print("\n Обучение Random Forest (Классификация)...")
    rf_clf = RandomForestClassifier(n_estimators=150, max_depth=10, class_weight='balanced', n_jobs=-1, random_state=42)
    rf_clf.fit(X_train_cls, y_train_risk)
    rf_pred = rf_clf.predict(X_test_cls)
    
    # ================= LSTM (Регрессор для RUL) =================
    print(" Обучение LSTM (Прогноз RUL)...")
    
    lstm_model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(SEQ_LENGTH, len(FEATURES))),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    lstm_model.fit(
        X_lstm_train, y_lstm_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=0
    )
    
    # Прогноз LSTM
    lstm_pred_scaled = lstm_model.predict(X_lstm_test, verbose=0)
    lstm_pred = scaler_rul.inverse_transform(lstm_pred_scaled).flatten()
    lstm_mae = mean_absolute_error(scaler_rul.inverse_transform(y_lstm_test.reshape(-1, 1)).flatten(), lstm_pred)

    # ================= NEURAL NETWORK (Классификация) =================
    print(" Обучение Neural Network (Классификация)...")
    mlp_clf = MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=300, random_state=42, verbose=False)
    mlp_clf.fit(X_train_cls_sc, y_train_risk)
    nn_proba = mlp_clf.predict_proba(X_test_cls_sc)[:, 1]
    
    # Оптимизация порога
    prec_curve, rec_curve, thresholds = precision_recall_curve(y_test_risk, nn_proba)
    TARGET_RECALL = 0.85
    safety_threshold = 0.5
    for p, r, t in zip(prec_curve, rec_curve, thresholds):
        if r >= TARGET_RECALL:
            safety_threshold = t
            break
    safety_threshold = max(safety_threshold, 0.08)
    print(f" Порог безопасности: {safety_threshold:.4f}")
    nn_pred = (nn_proba >= safety_threshold).astype(int)

    # ================= МЕТРИКИ =================
    def calc_metrics(y_true, y_pred, name):
        conf = confusion_matrix(y_true, y_pred)
        TN, FP, FN, TP = conf.ravel()
        rec = TP / (TP + FN) if (TP+FN)>0 else 0
        prec = TP / (TP + FP) if (TP+FP)>0 else 0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
        fnr = FN / (FN + TP) if (FN+TP)>0 else 0
        print(f" {name} -> Recall: {rec:.3f} | Precision: {prec:.3f} | FNR: {fnr:.3f}")
        return {'recall': rec, 'precision': prec, 'f1': f1, 'fnr': fnr, 'conf': conf}

    m_rf = calc_metrics(y_test_risk, rf_pred, "Random Forest (Clf)")
    m_nn = calc_metrics(y_test_risk, nn_pred, "Neural Network (Clf)")

    # ================= ГИБРИДНАЯ СИСТЕМА =================
    print("\n ЭТАП 3: Гибридная система")
    X_test_df = pd.DataFrame(X_test_cls, columns=FEATURES)
    hard_limit_risk = (
        (X_test_df['cno'] < 39.5) |          
        (X_test_df['timing_drift'] > 10.0) | 
        (X_test_df['pseudorange_res'] > 3.6)
    )
    hybrid_pred = (nn_pred == 1) | hard_limit_risk
    m_hybrid = calc_metrics(y_test_risk, hybrid_pred, "Hybrid System")

    # ================= ТАБЛИЦА =================
    print("\n" + "="*70)
    print(" ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print("="*70)
    print(f"{'Метрика':<25} {'RF (Clf)':<15} {'NN (Clf)':<15} {'LSTM (Reg)':<15} {'Hybrid':<10}")
    print("-"*70)
    print(f"{'MAE (RUL), часы':<25} {'-':<15} {'-':<15} {lstm_mae:<15.2f} {'-':<10}")
    print(f"{'Recall (Риск)':<25} {m_rf['recall']:<15.3f} {m_nn['recall']:<15.3f} {'-':<15} {m_hybrid['recall']:<10.3f}")
    print(f"{'Precision (Риск)':<25} {m_rf['precision']:<15.3f} {m_nn['precision']:<15.3f} {'-':<15} {m_hybrid['precision']:<10.3f}")
    print(f"{'F1-Score':<25} {m_rf['f1']:<15.3f} {m_nn['f1']:<15.3f} {'-':<15} {m_hybrid['f1']:<10.3f}")
    print(f"{'False Negative Rate':<25} {m_rf['fnr']:<15.3f} {m_nn['fnr']:<15.3f} {'-':<15} {m_hybrid['fnr']:<10.3f}")
    print("="*70)

    # ================= СОХРАНЕНИЕ МОДЕЛЕЙ =================
    print("\n Сохранение моделей...")
    joblib.dump(mlp_clf, "model_mlp.pkl")
    joblib.dump(scaler_cls, "scaler_mlp.pkl")
    joblib.dump(rf_clf, "model_rf_clf.pkl")
    joblib.dump(scaler_lstm, "scaler_lstm.pkl")
    joblib.dump(scaler_rul, "scaler_rul.pkl")
    lstm_model.save("model_lstm_rul.h5")
    
    print(" Сохранено:")
    print("   - model_mlp.pkl, scaler_mlp.pkl")
    print("   - model_rf_clf.pkl")
    print("   - model_lstm_rul.h5 (LSTM)")
    print("   - scaler_lstm.pkl, scaler_rul.pkl")

    # ================= ГРАФИКИ =================
    print("\n Построение финальных графиков...")
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 4, 1)
    sns.heatmap(m_rf['conf'], annot=True, fmt='d', cmap='Blues', xticklabels=['Норма', 'Риск'], yticklabels=['Норма', 'Риск'])
    plt.title('Random Forest (Clf)'); plt.ylabel('Факт'); plt.xlabel('Прогноз')

    plt.subplot(1, 4, 2)
    sns.heatmap(m_nn['conf'], annot=True, fmt='d', cmap='Greens', xticklabels=['Норма', 'Риск'], yticklabels=['Норма', 'Риск'])
    plt.title('Neural Network (Clf)'); plt.ylabel('Факт'); plt.xlabel('Прогноз')

    plt.subplot(1, 4, 3)
    sns.heatmap(m_hybrid['conf'], annot=True, fmt='d', cmap='Oranges', xticklabels=['Норма', 'Риск'], yticklabels=['Норма', 'Риск'])
    plt.title('Hybrid System'); plt.ylabel('Факт'); plt.xlabel('Прогноз')

    plt.subplot(1, 4, 4)
    PHYSICAL_FEATURES = ['temperature', 'voltage', 'cno', 'pseudorange_res', 'timing_drift', 'hours_since_maint', 'solar_activity']
    physical_indices = [FEATURES.index(f) for f in PHYSICAL_FEATURES]
    physical_importances = rf_clf.feature_importances_[physical_indices]
    sorted_idx = np.argsort(physical_importances)[::-1]
    
    RUSSIAN_NAMES = {
        'temperature': 'Температура', 'voltage': 'Напряжение', 'cno': 'C/N₀',
        'pseudorange_res': 'Ошибка псевдодальности', 'timing_drift': 'Дрейф времени',
        'hours_since_maint': 'Наработка с ТО', 'solar_activity': 'Солнечная активность'
    }
    russian_labels = [RUSSIAN_NAMES.get(f, f) for f in [PHYSICAL_FEATURES[i] for i in sorted_idx]]
    
    plt.barh(russian_labels, physical_importances[sorted_idx], color='skyblue')
    plt.title("Важность параметров (RF)"); plt.xlabel("Вклад"); plt.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("final_results.png", dpi=150)
    print(" final_results.png сохранён")
    return True

if __name__ == "__main__":
    t0 = time.time()
    if not run_cpp_generator(): sys.exit(1)
    if not run_ml_pipeline(): sys.exit(1)
    print(f"\n Готово! Время: {time.time()-t0:.2f} сек.")