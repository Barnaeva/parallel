import numpy as np
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 gen_matrix.py N")
        print("Example: python3 gen_matrix.py 200")
        sys.exit(1)
    
    n = int(sys.argv[1])
    
    print(f"Generating {n}x{n} matrices...")
    
    A = np.random.uniform(-10, 10, (n, n))
    B = np.random.uniform(-10, 10, (n, n))
    
    with open(f'test_A_{n}.txt', 'w') as f:
        f.write(f"{n}\n")
        for row in A:
            f.write(" ".join(f"{x:.6f}" for x in row) + "\n")
    
    with open(f'test_B_{n}.txt', 'w') as f:
        f.write(f"{n}\n")
        for row in B:
            f.write(" ".join(f"{x:.6f}" for x in row) + "\n")
    
    print(f"Done! Created test_A_{n}.txt and test_B_{n}.txt")

if __name__ == "__main__":
    main()