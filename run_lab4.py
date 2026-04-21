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

def run_opencl_multiply(n, prefix="test"):
    """Запускает OpenCL программу и возвращает время выполнения"""
    fileA = f"{prefix}_A_{n}.txt"
    fileB = f"{prefix}_B_{n}.txt"
    fileC = f"{prefix}_C_{n}.txt"
    
    result = subprocess.run(
        ["./lab4_opencl", fileA, fileB, fileC],
        capture_output=True,
        text=True
    )
    
    # Парсим вывод для получения времени и производительности
    time_ms = None
    ops = None
    perf = None
    
    for line in result.stdout.split('\n'):
        if 'Execution time:' in line:
            parts = line.split(':')
            if len(parts) >= 2:
                time_ms = float(parts[1].strip().split()[0])
        if 'Operations:' in line:
            parts = line.split(':')
            if len(parts) >= 2:
                ops = int(parts[1].strip())
        if 'Performance:' in line:
            parts = line.split(':')
            if len(parts) >= 2:
                perf = float(parts[1].strip().split()[0])
    
    print(result.stdout.strip())
    if result.stderr:
        print("  STDERR:", result.stderr.strip())
    
    return result.returncode == 0, time_ms, ops, perf

def save_results_to_csv(results, filename="results_opencl.csv"):
    """Сохраняет результаты в CSV файл"""
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['size', 'time_ms', 'operations', 'perf_M_ops', 'verification'])
        for row in results:
            writer.writerow(row)
    print(f"\n✅ Results saved to {filename}")

def main():
    # Размеры матриц для тестирования
    sizes = [200, 400, 800, 1200, 1600, 2000]
    
    # Список для хранения результатов
    results = []
    
    print("="*60)
    print("LABORATORY WORK #4: OpenCL MATRIX MULTIPLICATION")
    print("="*60)
    
    # Проверяем наличие программы
    if not os.path.exists("./lab4_opencl"):
        print("\nERROR: lab4_opencl not found!")
        print("Compile first: g++ lab4_opencl.cpp -o lab4_opencl -lOpenCL -std=c++11")
        sys.exit(1)
    
    for n in sizes:
        print(f"\n--- Testing N={n} ---")
        print("  Generating matrices...")
        generate_matrices(n)
        
        print("  Running OpenCL...")
        success, time_ms, ops, perf = run_opencl_multiply(n)
        
        if success and time_ms is not None:
            verification = verify_with_numpy(n)
            results.append([n, time_ms, ops, perf, verification])
        else:
            print(f"  ERROR: OpenCL program failed for N={n}")
    
    # Сохраняем результаты в CSV
    save_results_to_csv(results)
    
    # Вывод сводной таблицы
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'N':>6} | {'Time (ms)':>12} | {'Perf (M ops)':>14} | {'Verified':>10}")
    print("-"*70)
    for row in results:
        print(f"{row[0]:6d} | {row[1]:12.3f} | {row[3]:14.2f} | {row[4]:>10}")
    print("="*70)
    
    print("\n" + "="*60)
    print("LABORATORY WORK COMPLETED!")
    print(f"Results saved to results_opencl.csv")
    print("="*60)

if __name__ == "__main__":
    main()