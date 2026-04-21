import subprocess
import numpy as np
import os
import sys
import csv

def generate_matrices(n, prefix="test"):
    """Генерирует две случайные матрицы размера n x n"""
    A = np.random.uniform(-10, 10, (n, n))
    B = np.random.uniform(-10, 10, (n, n))
    
    with open(f"{prefix}_A_{n}.txt", "w") as f:
        f.write(f"{n}\n")
        for row in A:
            f.write(" ".join(f"{x:.6f}" for x in row) + "\n")
    
    with open(f"{prefix}_B_{n}.txt", "w") as f:
        f.write(f"{n}\n")
        for row in B:
            f.write(" ".join(f"{x:.6f}" for x in row) + "\n")
    
    return A, B

def verify_with_numpy(n, prefix="test"):
    """Сравнивает результат C++ с эталонным умножением NumPy"""
    A = np.loadtxt(f"{prefix}_A_{n}.txt", skiprows=1)
    B = np.loadtxt(f"{prefix}_B_{n}.txt", skiprows=1)
    C_cpp = np.loadtxt(f"{prefix}_C_{n}.txt", skiprows=1)
    C_ref = np.dot(A, B)
    
    if np.allclose(C_cpp, C_ref, rtol=1e-5, atol=1e-8):
        print(f"  VERIFICATION: PASSED for N={n}")
        return "PASSED"
    else:
        diff = np.max(np.abs(C_cpp - C_ref))
        print(f"  VERIFICATION: FAILED for N={n} (max diff={diff:.6e})")
        return "FAILED"

def run_mpi_multiply(n, num_processes, prefix="test"):
    """Запускает MPI программу и возвращает время выполнения"""
    fileA = f"{prefix}_A_{n}.txt"
    fileB = f"{prefix}_B_{n}.txt"
    fileC = f"{prefix}_C_{n}.txt"
    
    result = subprocess.run(
        ["mpiexec", "-n", str(num_processes), "./lab3_mpi", fileA, fileB, fileC],
        capture_output=True,
        text=True
    )
    
    # Парсим вывод для получения времени и производительности
    time_ms = None
    ops = None
    perf = None
    
    for line in result.stdout.split('\n'):
        if 'Execution time:' in line:
            time_ms = float(line.split(':')[1].strip().split()[0])
        if 'Operations:' in line:
            ops = int(line.split(':')[1].strip())
        if 'Performance:' in line:
            perf = float(line.split(':')[1].strip().split()[0])
    
    print(result.stdout.strip())
    if result.stderr:
        print("  STDERR:", result.stderr.strip())
    
    return result.returncode == 0, time_ms, ops, perf

def save_results_to_csv(results, filename="results_mpi.csv"):
    """Сохраняет результаты в CSV файл"""
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['size', 'processes', 'time_ms', 'operations', 'perf_M_ops', 'verification'])
        for row in results:
            writer.writerow(row)
    print(f"\n✅ Results saved to {filename}")

def main():
    # Размеры матриц для тестирования
    sizes = [200, 400, 800, 1200, 1600, 2000]
    # Количество процессов
    processes = [1, 2, 4, 8]
    
    # Список для хранения результатов
    results = []
    
    print("="*60)
    print("LABORATORY WORK #3: MPI MATRIX MULTIPLICATION")
    print("="*60)
    
    # Проверяем наличие программы
    if not os.path.exists("./lab3_mpi"):
        print("\nERROR: lab3_mpi not found!")
        print("Compile first: mpicxx -O2 lab3_mpi.cpp -o lab3_mpi")
        sys.exit(1)
    
    for n in sizes:
        print(f"\n--- Testing N={n} ---")
        print("  Generating matrices...")
        generate_matrices(n)
        
        for p in processes:
            print(f"\n  --- Processes={p} ---")
            print("    Running MPI...")
            success, time_ms, ops, perf = run_mpi_multiply(n, p)
            
            if success:
                verification = verify_with_numpy(n)
                results.append([n, p, time_ms, ops, perf, verification])
    
    # Сохраняем результаты в CSV
    save_results_to_csv(results)
    
    # Вывод сводной таблицы
    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    print(f"{'N':>6} | {'Proc':>4} | {'Time (ms)':>12} | {'Perf (M ops)':>14} | {'Verified':>8}")
    print("-"*60)
    for row in results:
        print(f"{row[0]:6d} | {row[1]:4d} | {row[2]:12.3f} | {row[4]:14.2f} | {row[5]:>8}")
    print("="*60)
    print("\nLABORATORY WORK COMPLETED!")

if __name__ == "__main__":
    main()