import matplotlib.pyplot as plt
import numpy as np
import os

def read_csv(filename):
    """Читает CSV файл без pandas"""
    sizes = []
    times = []
    perf = []
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            # Пропускаем заголовок
            for line in lines[1:]:
                if line.strip():
                    parts = line.strip().split(',')
                    if len(parts) >= 5:
                        sizes.append(int(parts[0]))
                        times.append(float(parts[2]))
                        perf.append(float(parts[4]))
        return sizes, times, perf
    except FileNotFoundError:
        print(f"Warning: {filename} not found!")
        return None, None, None

# Чтение данных из файлов
files = {
    1: 'res1.csv',
    2: 'res2.csv', 
    4: 'res4.csv',
    8: 'res8.csv',
    16: 'res16.csv'
}

data = {}
for proc, filename in files.items():
    sizes, times, perf = read_csv(filename)
    if sizes is not None:
        data[proc] = {
            'size': sizes,
            'time': times,
            'perf': perf
        }
        print(f"Loaded {filename}: {len(sizes)} entries")
    else:
        print(f"Warning: {filename} not found!")

if not data:
    print("No data files found! Please check filenames.")
    exit(1)

# Получаем размеры матриц (должны быть одинаковые во всех файлах)
sizes = list(data.values())[0]['size']

# Создаем словари для удобства
time_data = {proc: data[proc]['time'] for proc in data}
perf_data = {proc: data[proc]['perf'] for proc in data}

# Вычисляем ускорение (относительно 1 процесса)
speedup_data = {}
if 1 in data:
    base_time = data[1]['time']
    for proc in data:
        speedup_data[proc] = [base_time[i] / data[proc]['time'][i] for i in range(len(sizes))]

# Цвета и маркеры для графиков
colors = {1: 'blue', 2: 'red', 4: 'green', 8: 'purple', 16: 'orange'}
markers = {1: 'o', 2: 's', 4: '^', 8: 'D', 16: 'v'}
labels = {1: '1 процесс', 2: '2 процесса', 4: '4 процесса', 8: '8 процессов', 16: '16 процессов'}

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.size'] = 12

# ============================================================
# ГРАФИК 1: Время выполнения
# ============================================================
fig, ax = plt.subplots(figsize=(12, 7))
for proc in sorted(data.keys()):
    ax.plot(sizes, time_data[proc], 'o-', color=colors[proc], marker=markers[proc],
            linewidth=2, markersize=8, label=labels[proc])
ax.set_xlabel('Размер матрицы N', fontsize=14)
ax.set_ylabel('Время выполнения (мс)', fontsize=14)
ax.set_title('MPI: Время выполнения при разном количестве процессов', fontsize=16)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('mpi_time.png', dpi=150)
plt.show()

# ============================================================
# ГРАФИК 2: Ускорение (Speedup)
# ============================================================
if speedup_data:
    fig, ax = plt.subplots(figsize=(12, 7))
    for proc in sorted([p for p in speedup_data.keys() if p != 1]):
        ax.plot(sizes, speedup_data[proc], 'o-', color=colors[proc], marker=markers[proc],
                linewidth=2, markersize=8, label=labels[proc])
    # Идеальные линии
    for proc in [2, 4, 8, 16]:
        if proc in speedup_data:
            ax.plot(sizes, [proc] * len(sizes), 'k--', linewidth=1, alpha=0.4)
    ax.set_xlabel('Размер матрицы N', fontsize=14)
    ax.set_ylabel('Ускорение', fontsize=14)
    ax.set_title('MPI: Ускорение относительно 1 процесса', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('mpi_speedup.png', dpi=150)
    plt.show()

# ============================================================
# ГРАФИК 3: Эффективность параллелизации
# ============================================================
if speedup_data:
    fig, ax = plt.subplots(figsize=(12, 7))
    for proc in sorted([p for p in speedup_data.keys() if p != 1]):
        efficiency = [speedup_data[proc][i] / proc for i in range(len(sizes))]
        ax.plot(sizes, efficiency, 'o-', color=colors[proc], marker=markers[proc],
                linewidth=2, markersize=8, label=labels[proc])
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Идеальная эффективность')
    ax.set_xlabel('Размер матрицы N', fontsize=14)
    ax.set_ylabel('Эффективность', fontsize=14)
    ax.set_title('MPI: Эффективность параллелизации', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('mpi_efficiency.png', dpi=150)
    plt.show()

# ============================================================
# ГРАФИК 4: Производительность
# ============================================================
fig, ax = plt.subplots(figsize=(12, 7))
for proc in sorted(data.keys()):
    ax.plot(sizes, perf_data[proc], 'o-', color=colors[proc], marker=markers[proc],
            linewidth=2, markersize=8, label=labels[proc])
ax.set_xlabel('Размер матрицы N', fontsize=14)
ax.set_ylabel('Производительность (M ops/sec)', fontsize=14)
ax.set_title('MPI: Производительность при разном количестве процессов', fontsize=16)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('mpi_performance.png', dpi=150)
plt.show()

# ============================================================
# ГРАФИК 5: Логарифмический график времени
# ============================================================
fig, ax = plt.subplots(figsize=(12, 7))
for proc in sorted(data.keys()):
    ax.loglog(sizes, time_data[proc], 'o-', color=colors[proc], marker=markers[proc],
              linewidth=2, markersize=8, label=labels[proc])
# Теоретическая кривая O(N³)
if 1 in data:
    n_ref = sizes[0]
    t_ref = time_data[1][0]
    theoretical = t_ref * (np.array(sizes) / n_ref) ** 3
    ax.loglog(sizes, theoretical, 'k--', linewidth=1.5, alpha=0.5, label='Теория O(N³)')
ax.set_xlabel('Размер матрицы N (логарифм)', fontsize=14)
ax.set_ylabel('Время (мс) (логарифм)', fontsize=14)
ax.set_title('MPI: Логарифмический график времени выполнения', fontsize=16)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3, which='both')
plt.tight_layout()
plt.savefig('mpi_log_time.png', dpi=150)
plt.show()

# ============================================================
# ВЫВОД ТАБЛИЦ
# ============================================================
print("\n" + "="*90)
print("ТАБЛИЦА 1: ВРЕМЯ ВЫПОЛНЕНИЯ (мс)")
print("="*90)
header = f"{'N':>6} |"
for proc in sorted(data.keys()):
    header += f" {proc:>3} процессов |"
print(header)
print("-"*90)
for i, n in enumerate(sizes):
    row = f"{n:6d} |"
    for proc in sorted(data.keys()):
        row += f" {time_data[proc][i]:11.3f} |"
    print(row)

if speedup_data:
    print("\n" + "="*90)
    print("ТАБЛИЦА 2: УСКОРЕНИЕ (SPEEDUP)")
    print("="*90)
    header = f"{'N':>6} |"
    for proc in sorted([p for p in speedup_data.keys() if p != 1]):
        header += f" {proc:>3} процессов |"
    print(header)
    print("-"*90)
    for i, n in enumerate(sizes):
        row = f"{n:6d} |"
        for proc in sorted([p for p in speedup_data.keys() if p != 1]):
            row += f" {speedup_data[proc][i]:10.2f} |"
        print(row)

print("\n" + "="*90)
print("ТАБЛИЦА 3: ПРОИЗВОДИТЕЛЬНОСТЬ (M ops/sec)")
print("="*90)
header = f"{'N':>6} |"
for proc in sorted(data.keys()):
    header += f" {proc:>3} процессов |"
print(header)
print("-"*90)
for i, n in enumerate(sizes):
    row = f"{n:6d} |"
    for proc in sorted(data.keys()):
        row += f" {perf_data[proc][i]:11.2f} |"
    print(row)

print("\n" + "="*90)
print("Графики сохранены:")
print("  - mpi_time.png (время выполнения)")
print("  - mpi_speedup.png (ускорение)")
print("  - mpi_efficiency.png (эффективность)")
print("  - mpi_performance.png (производительность)")
print("  - mpi_log_time.png (логарифмический график)")
print("="*90)