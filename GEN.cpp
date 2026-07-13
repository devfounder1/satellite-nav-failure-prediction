// log_generator.cpp
// Компиляция: g++ -O2 -std=c++17 log_generator.cpp -o log_generator
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <string>
#include <iomanip>
#include <chrono>
#include <cmath>

struct LogEntry {
    double timestamp;           // Время в часах
    int    device_id;           // ID модуля
    double temperature;         // Температура, °C
    double voltage;             // Напряжение, В
    double cno;                 // C/N0, дБ·Гц
    double pseudorange_res;     // Ошибка псевдодальности, м
    double timing_drift;        // Дрейф времени, нс/ч
    double hours_since_maint;   // Часы с последнего ТО
    double solar_activity;      // Солнечная активность
    int    integrity_risk;      // Флаг риска (0/1)
    double rul;                 // Остаточный ресурс, ч
};

LogEntry generate_sample(int device_id, double time_hours, std::mt19937& rng) {
    LogEntry entry{};
    entry.timestamp = time_hours;
    entry.device_id = device_id;

    std::normal_distribution<double> temp_noise(45.0, 5.0);
    std::normal_distribution<double> volt_noise(5.0, 0.15);
    std::uniform_real_distribution<double> solar_dist(10.0, 80.0);
    std::normal_distribution<double> process_noise(0.0, 2.5);

    entry.temperature    = temp_noise(rng);
    entry.voltage        = volt_noise(rng);
    entry.solar_activity = solar_dist(rng);

    std::uniform_real_distribution<double> maint_offset(150.0, 700.0);
    entry.hours_since_maint = std::max(0.0, time_hours - maint_offset(rng));

    entry.cno             = 46.0 - 0.012 * time_hours + process_noise(rng);
    entry.pseudorange_res = 0.7 + 0.003 * time_hours + std::abs(process_noise(rng)) * 0.4;
    entry.timing_drift    = 5.5 + 0.0025 * time_hours + process_noise(rng) * 0.6;

    //  Рассчитываем индекс деградации БЕЗ std::clamp
    double deg_idx = 0.0;
    
    if (entry.cno < 38.0) {
        double val = (38.0 - entry.cno) / 8.0;
        deg_idx += (val < 0.0 ? 0.0 : (val > 1.0 ? 1.0 : val));
    }
    
    if (entry.pseudorange_res > 2.5) {
        double val = (entry.pseudorange_res - 2.5) / 4.0;
        deg_idx += (val < 0.0 ? 0.0 : (val > 1.0 ? 1.0 : val));
    }
    
    if (entry.timing_drift > 10.0) {
        double val = (entry.timing_drift - 10.0) / 8.0;
        deg_idx += (val < 0.0 ? 0.0 : (val > 1.0 ? 1.0 : val));
    }
    
    if (entry.temperature > 55.0 || entry.temperature < 25.0) {
        deg_idx += 0.2;
    }
    
    if (entry.voltage < 4.85 || entry.voltage > 5.15) {
        deg_idx += 0.15;
    }

    // Вероятностное назначение класса
    double risk_prob = 1.0 / (1.0 + std::exp(-3.0 * (deg_idx - 0.45)));
    std::bernoulli_distribution risk_dist(risk_prob);
    entry.integrity_risk = risk_dist(rng) ? 1 : 0;

    double base_rul = 1100.0;
    entry.rul = std::max(0.0, base_rul - time_hours * 0.6 - deg_idx * 200.0 + process_noise(rng) * 60.0);

    return entry;
}

int main() {
    const int NUM_DEVICES = 12;
    const double HOURS = 1000.0;
    const double DT = 1.0;
    const unsigned int SEED = 42;

    std::mt19937 rng(SEED);
    std::ofstream out("satellite_logs.csv");
    if (!out.is_open()) { std::cerr << "Ошибка файла!\n"; return 1; }

    out << "timestamp,device_id,temperature,voltage,cno,pseudorange_res,"
        << "timing_drift,hours_since_maint,solar_activity,integrity_risk,rul\n";

    auto start = std::chrono::high_resolution_clock::now();
    for (int dev = 0; dev < NUM_DEVICES; ++dev) {
        for (double t = 0.0; t < HOURS; t += DT) {
            LogEntry e = generate_sample(dev, t, rng);
            out << std::fixed << std::setprecision(3)
                << e.timestamp << "," << e.device_id << ","
                << e.temperature << "," << e.voltage << ","
                << e.cno << "," << e.pseudorange_res << ","
                << e.timing_drift << "," << e.hours_since_maint << ","
                << e.solar_activity << "," << e.integrity_risk << ","
                << e.rul << "\n";
        }
    }
    out.close();

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "[OK] Сгенерировано " << NUM_DEVICES * (HOURS / DT) 
              << " записей за " << std::chrono::duration<double>(end-start).count() << " с.\n";
    return 0;
}