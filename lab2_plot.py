import subprocess
import numpy as np
import os
import sys

def generate_matrices(n, prefix="test"):
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

def verify_with_numpy(n, prefix="test"):
    A = np.loadtxt(f"{prefix}_A_{n}.txt", skiprows=1)
    B = np.loadtxt(f"{prefix}_B_{n}.txt", skiprows=1)
    C_cpp = np.loadtxt(f"{prefix}_C_{n}.txt", skiprows=1)
    C_ref = np.dot(A, B)
    
    if np.allclose(C_cpp, C_ref, rtol=1e-5, atol=1e-8):
        print(f"  VERIFICATION: PASSED for N={n}")
        return True
    else:
        diff = np.max(np.abs(C_cpp - C_ref))
        print(f"  VERIFICATION: FAILED for N={n} (max diff={diff:.6e})")
        return False

def run_cpp_multiply(n, num_threads, prefix="test"):
    fileA = f"{prefix}_A_{n}.txt"
    fileB = f"{prefix}_B_{n}.txt"
    fileC = f"{prefix}_C_{n}.txt"
    exe_name = "lab2_omp.exe" if os.name == 'nt' else "./lab2_omp"
    
    result = subprocess.run(
        [exe_name, fileA, fileB, fileC, str(num_threads)],
        capture_output=True,
        text=True
    )
    print(result.stdout.strip())
    return result.returncode == 0

def main():
    # Меньше размеров для быстрого теста (можно увеличить)
    sizes = [200, 400, 800, 1200, 1600, 2000]
    threads = [1, 2, 4, 8]
    
    print("="*60)
    print("LABORATORY WORK #2: OpenMP MATRIX MULTIPLICATION")
    print("="*60)
    
    exe_name = "lab2_omp.exe" if os.name == 'nt' else "./lab2_omp"
    if not os.path.exists(exe_name):
        print("\nERROR: lab2_omp.exe not found!")
        print("Compile: g++ -O2 -fopenmp lab2_omp.cpp -o lab2_omp.exe")
        sys.exit(1)
    
    for n in sizes:
        print(f"\n--- Testing N={n} ---")
        print("  Generating matrices...")
        generate_matrices(n)
        
        for t in threads:
            print(f"\n  --- Threads={t} ---")
            print("    Running...")
            success = run_cpp_multiply(n, t)
            if success:
                verify_with_numpy(n)
    
    print("\n" + "="*60)
    print("LABORATORY WORK COMPLETED!")
    print("="*60)

if __name__ == "__main__":
    main()