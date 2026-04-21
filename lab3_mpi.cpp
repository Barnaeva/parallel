#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <mpi.h>

using namespace std;

// Чтение матрицы из файла
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

// Запись матрицы в файл
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

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc != 4) {
        if (rank == 0) {
            cerr << "Usage: mpiexec -n N " << argv[0] << " A.txt B.txt C.txt" << endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    string fileA = argv[1];
    string fileB = argv[2];
    string fileC = argv[3];
    
    int n;
    vector<vector<double>> A, B, C;
    
    // Только процесс 0 читает матрицы
    if (rank == 0) {
        A = readMatrix(fileA, n);
        B = readMatrix(fileB, n);
        C.resize(n, vector<double>(n, 0.0));
    }
    
    // Рассылаем размер матрицы всем процессам
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Подготовка для распределения строк
    int rows_per_proc = n / size;
    int remainder = n % size;
    
    // Определяем сколько строк получит каждый процесс
    vector<int> send_counts(size), displs(size);
    for (int i = 0; i < size; i++) {
        send_counts[i] = (i < remainder) ? (rows_per_proc + 1) : rows_per_proc;
        displs[i] = (i == 0) ? 0 : displs[i-1] + send_counts[i-1];
    }
    
    int local_rows = send_counts[rank];
    
    // Локальные матрицы
    vector<vector<double>> local_A(local_rows, vector<double>(n));
    vector<vector<double>> local_C(local_rows, vector<double>(n, 0.0));
    
    // Распределяем строки матрицы A по процессам
    if (rank == 0) {
        // Отправляем строки каждому процессу
        for (int p = 1; p < size; p++) {
            int start_row = displs[p];
            int rows = send_counts[p];
            for (int i = 0; i < rows; i++) {
                MPI_Send(A[start_row + i].data(), n, MPI_DOUBLE, p, i, MPI_COMM_WORLD);
            }
        }
        // Копируем строки для процесса 0
        for (int i = 0; i < local_rows; i++) {
            local_A[i] = A[displs[0] + i];
        }
    } else {
        // Принимаем строки матрицы A
        for (int i = 0; i < local_rows; i++) {
            MPI_Recv(local_A[i].data(), n, MPI_DOUBLE, 0, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    
    // Рассылаем матрицу B всем процессам
    if (rank == 0) {
        for (int p = 1; p < size; p++) {
            for (int i = 0; i < n; i++) {
                MPI_Send(B[i].data(), n, MPI_DOUBLE, p, i, MPI_COMM_WORLD);
            }
        }
    } else {
        B.resize(n, vector<double>(n));
        for (int i = 0; i < n; i++) {
            MPI_Recv(B[i].data(), n, MPI_DOUBLE, 0, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    
    // Замер времени
    double start_time = MPI_Wtime();
    
    // Локальное умножение
    for (int i = 0; i < local_rows; i++) {
        for (int k = 0; k < n; k++) {
            double aik = local_A[i][k];
            for (int j = 0; j < n; j++) {
                local_C[i][j] += aik * B[k][j];
            }
        }
    }
    
    double end_time = MPI_Wtime();
    double elapsed_ms = (end_time - start_time) * 1000.0;
    
    // Сбор результатов
    if (rank == 0) {
        // Копируем результат процесса 0
        for (int i = 0; i < local_rows; i++) {
            C[displs[0] + i] = local_C[i];
        }
        
        // Принимаем результаты от других процессов
        for (int p = 1; p < size; p++) {
            int start_row = displs[p];
            int rows = send_counts[p];
            for (int i = 0; i < rows; i++) {
                MPI_Recv(C[start_row + i].data(), n, MPI_DOUBLE, p, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        
        writeMatrix(fileC, C);
        
        long long ops = 2LL * n * n * n;
        double ops_per_sec = (ops / elapsed_ms) * 1000.0;
        
        cout << fixed << setprecision(3);
        cout << "Processes: " << size << endl;
        cout << "Execution time: " << elapsed_ms << " ms" << endl;
        cout << "Matrix size: " << n << " x " << n << endl;
        cout << "Operations: " << ops << endl;
        cout << "Performance: " << ops_per_sec / 1e6 << " M ops/sec" << endl;
        
    } else {
        // Отправляем результаты процессу 0
        for (int i = 0; i < local_rows; i++) {
            MPI_Send(local_C[i].data(), n, MPI_DOUBLE, 0, i, MPI_COMM_WORLD);
        }
    }
    
    MPI_Finalize();
    return 0;
}