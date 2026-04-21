import numpy as np
import matplotlib.pyplot as plt
import csv

def load_results_from_csv(filename="results_mpi.csv"):
    """Загружает результаты из CSV файла"""
    sizes = []
    processes = []
    times = []
    perf = []
    verification = []
    
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # пропускаем заголовок
        for row in reader:
            if len(row) >= 6:
                sizes.append(int(row[0]))
                processes.append(int(row[1]))
                times.append(float(row[2]))
                perf.append(float(row[4]))
                verification.append(row[5])
    
    # Преобразуем в массивы numpy
    sizes = np.array(sizes)
    processes = np.array(processes)
    times = np.array(times)
    perf = np.array(perf)
    
    return sizes, processes, times, perf, verification

def get_times_for_processes(sizes, processes, times, target_processes):
    """Получает время выполнения для указанного количества процессов"""
    result = []
    for p in target_processes:
        times_p = []
        for i, proc in enumerate(processes):
            if proc == p:
                times_p.append(times[i])
        result.append(times_p)
    return result

def main():
    # Загружаем данные из CSV
    try:
        sizes, processes, times, perf, verification = load_results_from_csv("results_mpi.csv")
        print("✅ Data loaded from results_mpi.csv")
    except FileNotFoundError:
        print("❌ results_mpi.csv not found!")
        print("Please run lab3_plot.py first to generate results")
        return
    
    # Уникальные размеры и количество процессов
    unique_sizes = np.unique(sizes)
    unique_processes = np.unique(processes)
    
    print(f"📊 Loaded: {len(unique_sizes)} sizes, {len(unique_processes)} process counts")
    print(f"   Sizes: {unique_sizes}")
    print(f"   Processes: {unique_processes}")
    
    # Получаем время для каждого количества процессов
    time_by_proc = []
    for p in unique_processes:
        times_p = []
        for n in unique_sizes:
            # Находим время для данного размера и количества процессов
            idx = np.where((sizes == n) & (processes == p))[0]
            if len(idx) > 0:
                times_p.append(times[idx[0]])
            else:
                times_p.append(0)
        time_by_proc.append(times_p)
    
    # Получаем производительность для каждого количества процессов
    perf_by_proc = []
    for p in unique_processes:
        perf_p = []
        for n in unique_sizes:
            idx = np.where((sizes == n) & (processes == p))[0]
            if len(idx) > 0:
                perf_p.append(perf[idx[0]])
            else:
                perf_p.append(0)
        perf_by_proc.append(perf_p)
    
    # Цвета и метки для графиков
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']
    labels = [f'{p} процессов' for p in unique_processes]
    
    # ============================================================
    # ГРАФИК 1: Время выполнения (обычный масштаб)
    # ============================================================
    plt.figure(figsize=(12, 7))
    for i, p in enumerate(unique_processes):
        plt.plot(unique_sizes, time_by_proc[i], 'o-', color=colors[i], 
                 linewidth=2, markersize=8, label=labels[i])
    
    plt.xlabel('Размер матрицы N', fontsize=12)
    plt.ylabel('Время выполнения (мс)', fontsize=12)
    plt.title('MPI: Время выполнения при разном количестве процессов', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('mpi_time.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # ============================================================
    # ГРАФИК 2: Время выполнения (ЛОГАРИФМИЧЕСКИЙ масштаб)
    # ============================================================
    plt.figure(figsize=(12, 7))
    for i, p in enumerate(unique_processes):
        plt.loglog(unique_sizes, time_by_proc[i], 'o-', color=colors[i], 
                   linewidth=2, markersize=8, label=labels[i])
    
    plt.xlabel('Размер матрицы N (логарифм)', fontsize=12)
    plt.ylabel('Время выполнения (мс) (логарифм)', fontsize=12)
    plt.title('MPI: Время выполнения (логарифмический масштаб)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3, which='both')
    plt.savefig('mpi_time_log.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # ============================================================
    # ГРАФИК 3: Ускорение (Speedup)
    # ============================================================
    # Базовое время для 1 процесса
    time_1proc = time_by_proc[0]
    
    plt.figure(figsize=(12, 7))
    for i, p in enumerate(unique_processes[1:], 1):  # пропускаем 1 процесс
        speedup = np.array(time_1proc) / np.array(time_by_proc[i])
        plt.plot(unique_sizes, speedup, 'o-', color=colors[i], 
                 linewidth=2, markersize=8, label=labels[i])
    
    # Идеальные линии ускорения
    for p in unique_processes[1:]:
        plt.plot(unique_sizes, [p] * len(unique_sizes), 'k--', 
                 linewidth=1, alpha=0.3, label=f'Идеал {p}' if p == unique_processes[1] else '')
    
    plt.xlabel('Размер матрицы N', fontsize=12)
    plt.ylabel('Ускорение (Speedup)', fontsize=12)
    plt.title('MPI: Ускорение относительно 1 процесса', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('mpi_speedup.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # ============================================================
    # ГРАФИК 4: Производительность
    # ============================================================
    plt.figure(figsize=(12, 7))
    for i, p in enumerate(unique_processes):
        plt.plot(unique_sizes, perf_by_proc[i], 'o-', color=colors[i], 
                 linewidth=2, markersize=8, label=labels[i])
    
    plt.xlabel('Размер матрицы N', fontsize=12)
    plt.ylabel('Производительность (M ops/sec)', fontsize=12)
    plt.title('MPI: Производительность при разном количестве процессов', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('mpi_performance.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # ============================================================
    # ГРАФИК 5: Ускорение только для больших матриц (от 800)
    # ============================================================
    mask = unique_sizes >= 800
    sizes_large = unique_sizes[mask]
    
    plt.figure(figsize=(10, 6))
    for i, p in enumerate(unique_processes[1:], 1):
        speedup = np.array(time_1proc[mask]) / np.array(time_by_proc[i][mask])
        plt.plot(sizes_large, speedup, 'o-', color=colors[i], 
                 linewidth=2, markersize=8, label=labels[i])
    
    for p in unique_processes[1:]:
        plt.plot(sizes_large, [p] * len(sizes_large), 'k--', 
                 linewidth=1, alpha=0.3)
    
    plt.xlabel('Размер матрицы N', fontsize=12)
    plt.ylabel('Ускорение (Speedup)', fontsize=12)
    plt.title('MPI: Ускорение (матрицы от 800 и выше)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('mpi_speedup_large.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # ============================================================
    # ВЫВОД ТАБЛИЦ
    # ============================================================
    print("\n" + "="*90)
    print("ТАБЛИЦА 1: ВРЕМЯ ВЫПОЛНЕНИЯ (мс)")
    print("="*90)
    header = f"{'N':>6} |"
    for p in unique_processes:
        header += f" {p:>3} процессов |"
    print(header)
    print("-"*90)
    
    for i, n in enumerate(unique_sizes):
        row = f"{n:6d} |"
        for j in range(len(unique_processes)):
            row += f" {time_by_proc[j][i]:11.3f} |"
        print(row)
    
    print("\n" + "="*90)
    print("ТАБЛИЦА 2: УСКОРЕНИЕ (SPEEDUP)")
    print("="*90)
    header = f"{'N':>6} |"
    for p in unique_processes[1:]:
        header += f" {p:>3} процессов |"
    print(header)
    print("-"*90)
    
    for i, n in enumerate(unique_sizes):
        row = f"{n:6d} |"
        for j in range(1, len(unique_processes)):
            speedup = time_by_proc[0][i] / time_by_proc[j][i]
            row += f" {speedup:11.3f} |"
        print(row)
    
    print("\n" + "="*90)
    print("ТАБЛИЦА 3: ПРОИЗВОДИТЕЛЬНОСТЬ (M ops/sec)")
    print("="*90)
    header = f"{'N':>6} |"
    for p in unique_processes:
        header += f" {p:>3} процессов |"
    print(header)
    print("-"*90)
    
    for i, n in enumerate(unique_sizes):
        row = f"{n:6d} |"
        for j in range(len(unique_processes)):
            row += f" {perf_by_proc[j][i]:11.2f} |"
        print(row)
    
    print("\n" + "="*90)
    print("Графики сохранены:")
    print("  - mpi_time.png")
    print("  - mpi_time_log.png (логарифмический масштаб)")
    print("  - mpi_speedup.png")
    print("  - mpi_performance.png")
    print("  - mpi_speedup_large.png (только большие матрицы)")
    print("="*90)

if __name__ == "__main__":
    main()