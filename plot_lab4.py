import matplotlib.pyplot as plt
import numpy as np
import csv

# Читаем данные из CSV файла
sizes = []
times = []
perf = []

with open('results_opencl.csv', 'r') as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        if len(row) >= 5:
            sizes.append(int(row[0]))
            times.append(float(row[1]))
            perf.append(float(row[3]))

# Данные из 1 лабораторной (CPU)
cpu_time = [8.506, 62.633, 535.114, 2006.269, 6161.557, 5967.706]
cpu_perf = [1881.07, 2043.65, 1913.61, 1722.60, 1329.53, 2681.10]

# Ускорение
speedup = [cpu_time[i] / times[i] for i in range(len(sizes))]

# Настройка стиля
plt.rcParams['font.size'] = 12

# ============================================================
# ГРАФИК 1: Сравнение времени выполнения CPU vs GPU
# ============================================================
plt.figure(figsize=(12, 7))
plt.plot(sizes, cpu_time, 'bo-', linewidth=2.5, markersize=10, label='CPU (1 поток)')
plt.plot(sizes, times, 'ro-', linewidth=2.5, markersize=10, label='GPU (OpenCL)')
plt.xlabel('Размер матрицы N', fontsize=14)
plt.ylabel('Время выполнения (мс)', fontsize=14)
plt.title('Сравнение времени выполнения: CPU vs GPU', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('cpu_vs_gpu_time.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# ГРАФИК 2: Производительность GPU
# ============================================================
plt.figure(figsize=(12, 7))
plt.plot(sizes, perf, 'purple', marker='s', linewidth=2.5, markersize=10)
plt.fill_between(sizes, perf, alpha=0.2, color='purple')
plt.xlabel('Размер матрицы N', fontsize=14)
plt.ylabel('Производительность (M ops/sec)', fontsize=14)
plt.title('Производительность GPU', fontsize=16)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('gpu_performance.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# ГРАФИК 3: Ускорение (ПРОСТОЙ И ПОНЯТНЫЙ)
# ============================================================
plt.figure(figsize=(12, 7))
plt.plot(sizes, speedup, 'go-', linewidth=2.5, markersize=10)
plt.axhline(y=1, color='red', linestyle='--', linewidth=2, label='Уровень CPU (1x)')
plt.xlabel('Размер матрицы N', fontsize=14)
plt.ylabel('Ускорение (раз)', fontsize=14)
plt.title('Ускорение GPU относительно CPU', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('gpu_speedup.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# ГРАФИК 4: Логарифмический график
# ============================================================
plt.figure(figsize=(12, 7))
plt.loglog(sizes, times, 'ro-', linewidth=2.5, markersize=10, label='GPU')
plt.loglog(sizes, cpu_time, 'bo-', linewidth=2.5, markersize=10, label='CPU')
n_ref = sizes[0]
t_ref = times[0]
theoretical = t_ref * (np.array(sizes) / n_ref) ** 3
plt.loglog(sizes, theoretical, 'g--', linewidth=1.5, alpha=0.7, label='Теория O(N³)')
plt.xlabel('Размер матрицы N (логарифм)', fontsize=14)
plt.ylabel('Время (мс) (логарифм)', fontsize=14)
plt.title('Логарифмический график времени выполнения', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3, which='both')
plt.tight_layout()
plt.savefig('log_time_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# ВЫВОД ТАБЛИЦЫ
# ============================================================
print("\n" + "="*70)
print("РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТОВ")
print("="*70)
print(f"{'N':>6} | {'CPU (мс)':>12} | {'GPU (мс)':>12} | {'Ускорение':>10}")
print("-"*70)
for i, n in enumerate(sizes):
    print(f"{n:6d} | {cpu_time[i]:12.3f} | {times[i]:12.3f} | {speedup[i]:10.2f}x")
print("="*70)

print("\n✅ Графики сохранены:")
print("  1. cpu_vs_gpu_time.png")
print("  2. gpu_performance.png")
print("  3. gpu_speedup.png")
print("  4. log_time_comparison.png")