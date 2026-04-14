#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>

using namespace std;
using namespace chrono;

vector<vector<double>> readMatrix(const string& filename, int& n) {
    ifstream fin(filename);
    if (!fin) {
        cerr << "ERROR: Cannot open " << filename << endl;
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

vector<vector<double>> multiplyMatrices(const vector<vector<double>>& A,
                                        const vector<vector<double>>& B) {
    int n = A.size();
    vector<vector<double>> C(n, vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < n; ++k) {
            double aik = A[i][k];
            for (int j = 0; j < n; ++j)
                C[i][j] += aik * B[k][j];
        }
    return C;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        cerr << "Usage: " << argv[0] << " A.txt B.txt C.txt [csv_output.csv]" << endl;
        return 1;
    }

    string fileA = argv[1];
    string fileB = argv[2];
    string fileC = argv[3];
    string csvFile = (argc >= 5) ? argv[4] : "results.csv";

    int n;
    vector<vector<double>> A = readMatrix(fileA, n);
    vector<vector<double>> B = readMatrix(fileB, n);

    auto start = high_resolution_clock::now();
    vector<vector<double>> C = multiplyMatrices(A, B);
    auto end = high_resolution_clock::now();
    double elapsed_ms = duration<double, milli>(end - start).count();

    writeMatrix(fileC, C);

    long long ops = 2LL * n * n * n;
    double ops_per_sec = (ops / elapsed_ms) * 1000.0;

    cout << fixed << setprecision(3);
    cout << "Execution time: " << elapsed_ms << " ms" << endl;
    cout << "Matrix size: " << n << " x " << n << endl;
    cout << "Operations: " << ops << endl;
    cout << "Performance: " << ops_per_sec / 1e6 << " M ops/sec" << endl;

    ofstream csvout(csvFile, ios::app);
    if (csvout.tellp() == 0) {
        csvout << "size,time_ms,operations,ops_per_sec_M\n";
    }
    csvout << n << "," << elapsed_ms << "," << ops << "," << ops_per_sec / 1e6 << "\n";
    csvout.close();

    return 0;
}