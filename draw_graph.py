import numpy as np
import matplotlib.pyplot as plt

# ВАШИ РЕАЛЬНЫЕ ДАННЫЕ из эксперимента
sizes = [200, 400, 800, 1200, 1600, 2000]

# Время выполнения (мс)
time_1 = [10.457, 87.268, 965.525, 7590.292, 21675.854, 88412.669]
time_2 = [10.764, 60.636, 531.369, 3448.671, 13310.825, 44971.032]
time_4 = [6.869, 40.430, 462.391, 2760.161, 8717.355, 24955.860]
time_8 = [6.986, 31.105, 616.824, 3493.070, 9084.148, 21791.660]

# Производительность (M ops/sec)
perf_1 = [1530.119, 1466.748, 1060.563, 455.318, 377.932, 180.970]
perf_2 = [1486.450, 2110.968, 1927.096, 1002.125, 615.439, 355.785]
perf_4 = [2329.339, 3165.989, 2214.575, 1252.101, 939.735, 641.132]
perf_8 = [2290.295, 4115.068, 1660.117, 989.388, 901.791, 734.226]

# ============================================================
# ГРАФИК 1: Время выполнения
# ============================================================
plt.figure(figsize=(12, 7))
plt.plot(sizes, time_1, 'bo-', linewidth=2, markersize=8, label='1 поток')
plt.plot(sizes, time_2, 'ro-', linewidth=2, markersize=8, label='2 потока')
plt.plot(sizes, time_4, 'go-', linewidth=2, markersize=8, label='4 потока')
plt.plot(sizes, time_8, 'mo-', linewidth=2, markersize=8, label='8 потоков')

plt.xlabel('Размер матрицы N', fontsize=12)
plt.ylabel('Время выполнения (мс)', fontsize=12)
plt.title('OpenMP: Время выполнения при разном количестве потоков', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('omp_time.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# ГРАФИК 2: Ускорение (Speedup)
# ============================================================
speedup_2 = np.array(time_1) / np.array(time_2)
speedup_4 = np.array(time_1) / np.array(time_4)
speedup_8 = np.array(time_1) / np.array(time_8)

plt.figure(figsize=(12, 7))
plt.plot(sizes, speedup_2, 'ro-', linewidth=2, markersize=8, label='2 потока')
plt.plot(sizes, speedup_4, 'go-', linewidth=2, markersize=8, label='4 потока')
plt.plot(sizes, speedup_8, 'mo-', linewidth=2, markersize=8, label='8 потоков')
plt.plot(sizes, [2]*len(sizes), 'k--', linewidth=1, alpha=0.5, label='Идеал 2')
plt.plot(sizes, [4]*len(sizes), 'k--', linewidth=1, alpha=0.5, label='Идеал 4')
plt.plot(sizes, [8]*len(sizes), 'k--', linewidth=1, alpha=0.5, label='Идеал 8')

plt.xlabel('Размер матрицы N', fontsize=12)
plt.ylabel('Ускорение (Speedup)', fontsize=12)
plt.title('OpenMP: Ускорение относительно 1 потока', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('omp_speedup.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# ГРАФИК 3: Эффективность параллелизации
# ============================================================
efficiency_2 = speedup_2 / 2
efficiency_4 = speedup_4 / 4
efficiency_8 = speedup_8 / 8

plt.figure(figsize=(12, 7))
plt.plot(sizes, efficiency_2, 'ro-', linewidth=2, markersize=8, label='2 потока')
plt.plot(sizes, efficiency_4, 'go-', linewidth=2, markersize=8, label='4 потока')
plt.plot(sizes, efficiency_8, 'mo-', linewidth=2, markersize=8, label='8 потоков')
plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Идеальная эффективность')

plt.xlabel('Размер матрицы N', fontsize=12)
plt.ylabel('Эффективность', fontsize=12)
plt.title('OpenMP: Эффективность параллелизации', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('omp_efficiency.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# ГРАФИК 4: Производительность
# ============================================================
plt.figure(figsize=(12, 7))
plt.plot(sizes, perf_1, 'bo-', linewidth=2, markersize=8, label='1 поток')
plt.plot(sizes, perf_2, 'ro-', linewidth=2, markersize=8, label='2 потока')
plt.plot(sizes, perf_4, 'go-', linewidth=2, markersize=8, label='4 потока')
plt.plot(sizes, perf_8, 'mo-', linewidth=2, markersize=8, label='8 потоков')

plt.xlabel('Размер матрицы N', fontsize=12)
plt.ylabel('Производительность (M ops/sec)', fontsize=12)
plt.title('OpenMP: Производительность при разном количестве потоков', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('omp_performance.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# ВЫВОД ТАБЛИЦ
# ============================================================
print("\n" + "="*90)
print("ТАБЛИЦА 1: ВРЕМЯ ВЫПОЛНЕНИЯ (мс)")
print("="*90)
print(f"{'N':>6} | {'1 поток':>12} | {'2 потока':>12} | {'4 потока':>12} | {'8 потоков':>12}")
print("-"*90)
for i, n in enumerate(sizes):
    print(f"{n:6d} | {time_1[i]:12.3f} | {time_2[i]:12.3f} | {time_4[i]:12.3f} | {time_8[i]:12.3f}")

print("\n" + "="*90)
print("ТАБЛИЦА 2: УСКОРЕНИЕ (SPEEDUP)")
print("="*90)
print(f"{'N':>6} | {'2 потока':>12} | {'4 потока':>12} | {'8 потоков':>12}")
print("-"*90)
for i, n in enumerate(sizes):
    print(f"{n:6d} | {speedup_2[i]:12.3f} | {speedup_4[i]:12.3f} | {speedup_8[i]:12.3f}")

print("\n" + "="*90)
print("ТАБЛИЦА 3: ЭФФЕКТИВНОСТЬ")
print("="*90)
print(f"{'N':>6} | {'2 потока':>12} | {'4 потока':>12} | {'8 потоков':>12}")
print("-"*90)
for i, n in enumerate(sizes):
    print(f"{n:6d} | {efficiency_2[i]:12.3f} | {efficiency_4[i]:12.3f} | {efficiency_8[i]:12.3f}")

print("\n" + "="*90)
print("ТАБЛИЦА 4: ПРОИЗВОДИТЕЛЬНОСТЬ (M ops/sec)")
print("="*90)
print(f"{'N':>6} | {'1 поток':>12} | {'2 потока':>12} | {'4 потока':>12} | {'8 потоков':>12}")
print("-"*90)
for i, n in enumerate(sizes):
    print(f"{n:6d} | {perf_1[i]:12.3f} | {perf_2[i]:12.3f} | {perf_4[i]:12.3f} | {perf_8[i]:12.3f}")

print("\n" + "="*90)
print("Графики сохранены:")
print("  - omp_time.png")
print("  - omp_speedup.png")
print("  - omp_efficiency.png")
print("  - omp_performance.png")
print("="*90)