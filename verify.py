import numpy as np
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 verify.py N")
        print("Example: python3 verify.py 200")
        sys.exit(1)
    
    n = int(sys.argv[1])
    
    A = np.loadtxt(f'test_A_{n}.txt', skiprows=1)
    B = np.loadtxt(f'test_B_{n}.txt', skiprows=1)
    C = np.loadtxt(f'test_C_{n}.txt', skiprows=1)
    
    C_ref = np.dot(A, B)
    
    if np.allclose(C, C_ref, rtol=1e-5, atol=1e-8):
        print(f"VERIFICATION: PASSED for N={n} ✓")
    else:
        diff = np.max(np.abs(C - C_ref))
        print(f"VERIFICATION: FAILED for N={n} (max diff={diff:.6e})")

if __name__ == "__main__":
    main()