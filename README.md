# Satellite Navigation Equipment Failure Prediction & Integrity Control

A hybrid Machine Learning system for predicting Remaining Useful Life (RUL) and integrity risks in aviation satellite landing systems. The project combines high-performance C++ data generation with a Python-based deep learning pipeline (LSTM + MLP) to transition from calendar-based maintenance to condition-based predictive maintenance.

> *This project was developed as part of my Specialist Diploma research at SUAI (Saint Petersburg State University of Aerospace Instrumentation).*

## Key Achievements & Impact
Based on modeling of 12 navigation modules over 1000+ hours of operation:
- +6.2% increase in failure detection probability (up to 98.7%).
- -73% reduction in system reaction time (from 45 min to 12 min).
- -75% reduction in false alarms compared to static threshold methods.
- 26% operational cost savings over 5 years (payback period < 1 year).

## Tech Stack
- Data Generation: C++17 (`<random>`, `std::mt19937`, high-performance I/O)
- Machine Learning: Python, TensorFlow/Keras (LSTM, MLP), Scikit-Learn (Random Forest)
- Data Processing: Pandas, NumPy, Scikit-Learn (StandardScaler, MinMaxScaler)
- Visualization: Matplotlib, Seaborn
- Tools: Git, CMake/Make, Jupyter/VS Code

## System Architecture
The project is divided into three modular components:

1. `GEN.cpp` (C++ Data Simulator)  
   Generates realistic, physics-based synthetic degradation data. Simulates 12 devices over 1000 hours, modeling the decay of C/N₀ (Carrier-to-Noise density), pseudorange residuals, and timing drift using the Mersenne Twister engine. Outputs a structured CSV dataset in milliseconds.

2. `main_pipeline.py` (ML Training & Hybrid Logic)  
   - **Feature Engineering**: Creates rolling averages, anomaly scores, and thermal stress metrics.
   - **Modeling**: Trains a Random Forest (baseline), an MLP Classifier (risk probability), and an LSTM Regressor (RUL forecasting).
   - **Hybrid Decision System**: Combines neural network probabilistic outputs with hard physical limits (e.g., `C/N₀ < 39.5`) to minimize False Negative Rates (FNR).

3. `demo_case_study.py` (Inference & Visualization) 
   Loads the trained models and generates professional, 3-panel analytical reports for specific devices. Visualizes raw signal trends, risk probability over time, and RUL forecasts, highlighting early warning triggers and calculated "time saved".

### Прогнозирование остаточного ресурса (RUL) и вероятности отказа
![Case Study Analysis](images/case_studies/module_0_case_PRO.png)

### Сравнение моделей и важность признаков
![Model Comparison](images/final_results.png)
