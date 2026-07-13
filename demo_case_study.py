# demo_case_study.py
# Профессиональная визуализация кейсов (без эмодзи, исправлена вёрстка)
import os
import sys
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings

# Скрываем технические предупреждения (TensorFlow)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")

# Настройка шрифтов
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Calibri', 'Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

def ensure_environment_ready():
    if not os.path.exists("satellite_logs.csv"):
        print("[SYS] Файл данных не найден. Запускаю генератор (GEN.exe)...")
        if os.path.exists("GEN.exe"):
            subprocess.run(["GEN.exe"])
            print("[SYS] satellite_logs.csv создан.")
        else:
            print("[ERROR] GEN.exe не найден.")
            sys.exit(1)

    required_files = ["model_mlp.pkl", "scaler_mlp.pkl", "model_lstm_rul.h5", "scaler_lstm.pkl", "scaler_rul.pkl"]
    missing = [f for f in required_files if not os.path.exists(f)]
    
    if missing:
        print(f"[WARN] Не найдены файлы: {', '.join(missing)}")
        print("[SYS] Запускаю обучение (main_pipeline.py)...")
        result = subprocess.run([sys.executable, "main_pipeline.py"])
        if result.returncode != 0:
            print("[ERROR] Ошибка обучения.")
            sys.exit(1)
        print("[SYS] Модели обучены.")
    else:
        print("[SYS] Все компоненты готовы.")

def prepare_features(data):
    data = data.copy()
    data['cno_ma5'] = data['cno'].rolling(5, min_periods=1).mean()
    data['cno_anomaly'] = data['cno'] - data['cno_ma5']
    data['temp_stress'] = (data['temperature'] - 40.0).abs()
    data['drift_rate'] = data['timing_drift'].diff().fillna(0)
    features = ['temperature', 'voltage', 'cno', 'pseudorange_res', 
                'timing_drift', 'hours_since_maint', 'solar_activity',
                'cno_ma5', 'cno_anomaly', 'temp_stress', 'drift_rate']
    return data[features]

def create_single_sequence(data, scaler_lstm, seq_length=50):
    scaled_data = scaler_lstm.transform(data)
    if len(scaled_data) < seq_length:
        padding = np.zeros((seq_length - len(scaled_data), scaled_data.shape[1]))
        return np.vstack([padding, scaled_data])[np.newaxis, ...]
    return scaled_data[-seq_length:][np.newaxis, ...]

def analyze_module(device_id, df, mlp_model, scaler_cls, lstm_model, scaler_lstm, scaler_rul, output_dir="case_studies"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    device_data = df[df['device_id'] == device_id].sort_values('timestamp').reset_index(drop=True)
    X_demo = prepare_features(device_data)
    
    X_demo_scaled = scaler_cls.transform(X_demo)
    nn_probs = mlp_model.predict_proba(X_demo_scaled)[:, 1]
    
    SEQ_LENGTH = 50
    rul_predictions = []
    for i in range(len(device_data)):
        seq = create_single_sequence(X_demo.iloc[:i+1], scaler_lstm, SEQ_LENGTH) if i < SEQ_LENGTH else create_single_sequence(X_demo.iloc[i-SEQ_LENGTH+1:i+1], scaler_lstm, SEQ_LENGTH)
        rul_pred = scaler_rul.inverse_transform([[lstm_model.predict(seq, verbose=0)[0, 0]]])[0, 0]
        rul_predictions.append(rul_pred)
    
    device_data['nn_prob'] = nn_probs
    device_data['rul_pred'] = rul_predictions
    
    ALARM_THRESHOLD = 0.10
    alarm_indices = [i for i, p in enumerate(nn_probs) if p >= ALARM_THRESHOLD]
    first_alarm_idx = alarm_indices[0] if alarm_indices else -1
    
    failure_mask = device_data['cno'] < 30.0
    fail_indices = device_data[failure_mask].index
    first_fail_idx = fail_indices.min() if len(fail_indices) > 0 else len(device_data) - 1
    
    current_time = device_data['timestamp'].iloc[-1]
    current_rul = rul_predictions[-1]
    time_saved = 0
    alarm_time = None
    fail_time = device_data.loc[first_fail_idx, 'timestamp']
    
    if first_alarm_idx != -1 and first_alarm_idx < first_fail_idx:
        alarm_time = device_data.loc[first_alarm_idx, 'timestamp']
        time_saved = fail_time - alarm_time

    # ================= ВИЗУАЛИЗАЦИЯ =================
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f'[SYS] Анализ модуля #{device_id}: Прогнозирование отказа', fontsize=15, fontweight='bold', y=0.97)
    
    # --- Панель 1: Качество сигнала (C/N0) ---
    ax1.axhspan(30, 35, alpha=0.15, color='red', label='Critical Zone')
    ax1.axhspan(35, 40, alpha=0.15, color='orange', label='Warning Zone')
    ax1.axhspan(40, 55, alpha=0.1, color='green', label='Normal Zone')
    
    ax1.plot(device_data['timestamp'], device_data['cno'], 'b-', linewidth=0.5, alpha=0.4, label='C/N0 (raw)')
    ax1.plot(device_data['timestamp'], pd.Series(device_data['cno']).rolling(20, min_periods=1).mean(), 'b-', linewidth=2.5, label='C/N0 (trend)')
    ax1.axhline(30, color='red', linestyle='--', linewidth=2, label='Failure Threshold (<30)')
    ax1.axhline(39.5, color='orange', linestyle=':', linewidth=1.5, label='Pre-Alarm (<39.5)')
    ax1.set_ylabel('C/N0, dB-Hz', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper right', fontsize=9)
    
    # --- Панель 2: Вероятность отказа ---
    smooth_probs = pd.Series(nn_probs).rolling(50, min_periods=1).mean()
    ax2.plot(device_data['timestamp'], smooth_probs, 'r-', linewidth=2.5, label='Failure Probability (trend)')
    ax2.fill_between(device_data['timestamp'], smooth_probs, alpha=0.2, color='red')
    ax2.axhline(ALARM_THRESHOLD, color='orange', linestyle=':', linewidth=2.5, label=f'Alarm Threshold ({ALARM_THRESHOLD})')
    ax2.set_ylabel('Risk Probability', color='r', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='upper left', fontsize=9)
    
    # --- Панель 3: Остаточный ресурс (RUL) ---
    smooth_rul = pd.Series(rul_predictions).rolling(20, min_periods=1).mean()
    ax3.plot(device_data['timestamp'], smooth_rul, 'g-', linewidth=2.5, label='RUL Forecast (LSTM)')
    ax3.fill_between(device_data['timestamp'], smooth_rul, alpha=0.2, color='green')
    ax3.set_ylabel('Remaining Life, hours', color='g', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Operating Time, hours', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.legend(loc='upper right', fontsize=9)
    
    # --- Аннотации событий ---
    if first_alarm_idx != -1:
        for ax in [ax1, ax2, ax3]:
            ax.axvline(alarm_time, color='orange', linewidth=2, alpha=0.8, linestyle='-.')
        ax2.text(alarm_time, 0.85, f'[ALARM]\n{alarm_time:.0f} h', ha='center', color='orange', fontweight='bold', fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.9, edgecolor='orange', boxstyle='round,pad=0.3'))
    
    ax1.axvline(fail_time, color='red', linewidth=2.5, alpha=0.9, linestyle='--')
    ax1.text(fail_time, 48, f'[FAILURE]\nC/N0 < 30\n{fail_time:.0f} h', ha='center', color='red', fontweight='bold', fontsize=10,
             bbox=dict(facecolor='white', alpha=0.95, edgecolor='red', boxstyle='round,pad=0.3'))
    
    # --- Инфоблок (чистый ASCII, безопасное позиционирование) ---
    info_lines = [
        "[STATUS] MODULE #" + str(device_id),
        "-------------------",
        f"Operating Time : {current_time:.0f} h",
        f"RUL Forecast   : {current_rul:.0f} h",
        f"Plan Replace By: {current_time + current_rul:.0f} h",
        "-------------------",
        f"Alarm Trigger  : {alarm_time:.0f} h" if alarm_time else "Alarm Trigger  : None",
        f"Time Saved     : {time_saved:.0f} h ({time_saved/24:.1f} d)" if time_saved > 0 else "Time Saved     : N/A"
    ]
    info_text = "\n".join(info_lines)
    
    ax1.text(0.015, 0.98, info_text, transform=ax1.transAxes, fontsize=8.5, va='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.95, edgecolor='gray', linewidth=1.5))
    
    # Фикс вёрстки: оставляем место под заголовок и подписи
    plt.subplots_adjust(hspace=0.25, top=0.92, bottom=0.08, left=0.1, right=0.95)
    
    filename = f"{output_dir}/module_{device_id}_case_PRO.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    return {'device_id': device_id, 'time_saved': time_saved, 'current_rul': current_rul, 'filename': filename}

def run_demo():
    print("\n" + "="*70)
    print("[DEMO] PROFESSIONAL ANALYSIS: Detailed Case Studies")
    print("="*70)
    
    print("[LOAD] Loading models...")
    mlp_model = joblib.load("model_mlp.pkl")
    scaler_cls = joblib.load("scaler_mlp.pkl")
    scaler_lstm = joblib.load("scaler_lstm.pkl")
    scaler_rul = joblib.load("scaler_rul.pkl")
    
    from tensorflow.keras.models import load_model
    lstm_model = load_model("model_lstm_rul.h5", compile=False)
    print("[OK] Models loaded successfully.")
    
    df = pd.read_csv("satellite_logs.csv")
    risky_devices = df[df['integrity_risk'] == 1]['device_id'].unique()
    if len(risky_devices) == 0: 
        print("[ERROR] No failed modules found in data!")
        sys.exit(1)
    
    print(f"\n[SCAN] Found {len(risky_devices)} modules for analysis.")
    print("="*70)
    
    results = []
    for device_id in risky_devices:
        print(f"\n[PROC] Analyzing module #{device_id}...")
        result = analyze_module(device_id, df, mlp_model, scaler_cls, lstm_model, scaler_lstm, scaler_rul)
        results.append(result)
        status = "[OK] EARLY WARNING" if result['time_saved'] > 0 else "[WARN] TRIGGERED DURING DEGRADATION"
        print(f"   {status}: saved {result['time_saved']:.1f} h | RUL: {result['current_rul']:.1f} h")
    
    print("\n" + "="*70)
    print("[SUMMARY] AGGREGATED STATISTICS")
    print("="*70)
    total = len(results)
    early = sum(1 for r in results if r['time_saved'] > 0)
    avg_saved = np.mean([r['time_saved'] for r in results if r['time_saved'] > 0]) if early > 0 else 0
    print(f"Total modules analyzed : {total}")
    print(f"Early warnings         : {early} ({100*early/total:.1f}%)")
    print(f"Average time saved     : {avg_saved:.1f} h ({avg_saved/24:.1f} days)")
    print("="*70)
    print(f"\n[SAVE] All graphs saved to: case_studies/")
    print("[DONE] Analysis complete.")

if __name__ == "__main__":
    ensure_environment_ready()
    run_demo()