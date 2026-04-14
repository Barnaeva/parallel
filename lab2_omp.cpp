#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <omp.h>

using namespace std;
using namespace chrono;

vector<vector<double>> readMatrix(const string& filename, int& n) {
    ifstream fin(filename);
    if (!fin) {
        cerr << "Error: cannot open " << filename << endl;
        exit(1);
    }
    fin >> n;
    vector<vector<double>> mat(n, vector<double>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            fin >> mat[i][j];
    fin.close();
    return mat;
}

void writeMatrix(const string& filename, const vector<vector<double>>& mat) {
    ofstream fout(filename);
    int n = mat.size();
    fout << n << "\n";
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j)
            fout << mat[i][j] << " ";
        fout << "\n";
    }
    fout.close();
}

// Параллельное умножение (добавлен OpenMP)
vector<vector<double>> multiplyMatrices(const vector<vector<double>>& A,
                                        const vector<vector<double>>& B,
                                        int num_threads) {
    int n = A.size();
    vector<vector<double>> C(n, vector<double>(n, 0.0));
    
    omp_set_num_threads(num_threads);
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
    
    return C;
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        cerr << "Usage: " << argv[0] << " A.txt B.txt C.txt num_threads" << endl;
        return 1;
    }

    string fileA = argv[1];
    string fileB = argv[2];
    string fileC = argv[3];
    int num_threads = atoi(argv[4]);

    int n;
    vector<vector<double>> A = readMatrix(fileA, n);
    vector<vector<double>> B = readMatrix(fileB, n);

    auto start = high_resolution_clock::now();
    vector<vector<double>> C = multiplyMatrices(A, B, num_threads);
    auto end = high_resolution_clock::now();
    double elapsed_ms = duration<double, milli>(end - start).count();

    writeMatrix(fileC, C);

    long long ops = 2LL * n * n * n;
    double ops_per_sec = (ops / elapsed_ms) * 1000.0;

    cout << fixed << setprecision(3);
    cout << "Threads: " << num_threads << endl;
    cout << "Execution time: " << elapsed_ms << " ms" << endl;
    cout << "Matrix size: " << n << " x " << n << endl;
    cout << "Operations: " << ops << endl;
    cout << "Performance: " << ops_per_sec / 1e6 << " M ops/sec" << endl;

    return 0;
}