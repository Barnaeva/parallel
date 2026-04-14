import subprocess
import numpy as np
import os
import sys

# ============================================================
# 1. ГЕНЕРАЦИЯ ТЕСТОВЫХ МАТРИЦ
# ============================================================
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

# ============================================================
# 2. ВЕРИФИКАЦИЯ ЧЕРЕЗ NUMPY
# ============================================================
def verify_with_numpy(n, prefix="test"):
    """Сравнивает результат C++ с эталонным умножением NumPy"""
    # Читаем матрицы
    A = np.loadtxt(f"{prefix}_A_{n}.txt", skiprows=1)
    B = np.loadtxt(f"{prefix}_B_{n}.txt", skiprows=1)
    C_cpp = np.loadtxt(f"{prefix}_C_{n}.txt", skiprows=1)
    
    # Эталонное умножение
    C_ref = np.dot(A, B)
    
    # Сравнение
    if np.allclose(C_cpp, C_ref, rtol=1e-5, atol=1e-8):
        print(f"  VERIFICATION: PASSED for N={n}")
        return True
    else:
        diff = np.max(np.abs(C_cpp - C_ref))
        print(f"  VERIFICATION: FAILED for N={n} (max diff={diff:.6e})")
        return False

# ============================================================
# 3. ЗАПУСК C++ ПРОГРАММЫ
# ============================================================
def run_cpp_multiply(n, prefix="test", csv_file="results.csv"):
    """Запускает скомпилированную C++ программу"""
    fileA = f"{prefix}_A_{n}.txt"
    fileB = f"{prefix}_B_{n}.txt"
    fileC = f"{prefix}_C_{n}.txt"
    
    # Определяем имя исполняемого файла в зависимости от ОС
    exe_name = "./lab1" if os.name != 'nt' else "lab1.exe"
    
    result = subprocess.run(
        [exe_name, fileA, fileB, fileC, csv_file],
        capture_output=True,
        text=True
    )
    print(result.stdout.strip())
    return result.returncode == 0

# ============================================================
# 4. ОСНОВНАЯ ФУНКЦИЯ
# ============================================================
def main():
    # Размеры матриц для тестирования (согласно заданию)
    sizes = [200, 400, 800, 1200, 1600, 2000]
    
    # Удаляем старый CSV
    if os.path.exists("results.csv"):
        os.remove("results.csv")
    
    print("="*60)
    print("LABORATORY WORK #1: MATRIX MULTIPLICATION")
    print("="*60)
    
    # Проверяем наличие скомпилированной программы
    exe_name = "./lab1" if os.name != 'nt' else "lab1.exe"
    if not os.path.exists(exe_name):
        print("\nERROR: lab1 executable not found!")
        print("Please compile first: g++ -O2 lab1.cpp -o lab1")
        sys.exit(1)
    
    # Основной цикл экспериментов
    for n in sizes:
        print(f"\n--- Testing N={n} ---")
        
        # Генерация матриц
        print("  Generating matrices...")
        generate_matrices(n)
        
        # Запуск C++ программы
        print("  Running C++ multiplication...")
        success = run_cpp_multiply(n)
        
        if not success:
            print(f"  ERROR: C++ program failed for N={n}")
            continue
        
        # Верификация
        verify_with_numpy(n)
    
   

if __name__ == "__main__":
    main()