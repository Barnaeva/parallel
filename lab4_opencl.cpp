#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <ctime>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

using namespace std;
using namespace chrono;

void checkError(cl_int err, const string &name) {
    if (err != CL_SUCCESS) {
        cerr << "OpenCL Error: " << name << " (" << err << ")" << endl;
        exit(EXIT_FAILURE);
    }
}

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

// Ядро OpenCL с использованием double
const char* kernelSource = R"(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void matrixMul(__global const double* A,
                        __global const double* B,
                        __global double* C,
                        int n) {
    int row = get_global_id(0);
    int col = get_global_id(1);

    if (row < n && col < n) {
        double sum = 0.0;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}
)";

int main(int argc, char* argv[]) {
    if (argc != 4) {
        cerr << "Usage: " << argv[0] << " A.txt B.txt C.txt" << endl;
        return 1;
    }

    string fileA = argv[1];
    string fileB = argv[2];
    string fileC = argv[3];

    int n;
    vector<vector<double>> A = readMatrix(fileA, n);
    vector<vector<double>> B = readMatrix(fileB, n);
    vector<vector<double>> C(n, vector<double>(n, 0.0));

    vector<double> flatA(n * n);
    vector<double> flatB(n * n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            flatA[i * n + j] = A[i][j];
            flatB[i * n + j] = B[i][j];
        }
    }
    vector<double> flatC(n * n, 0.0);

    cl_int err;
    
    cl_platform_id platform;
    err = clGetPlatformIDs(1, &platform, nullptr);
    checkError(err, "Getting platform ID");

    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    if (err == CL_DEVICE_NOT_FOUND) {
        cerr << "No GPU found, trying CPU..." << endl;
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, nullptr);
    }
    checkError(err, "Getting device ID");

    char deviceName[256];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
    cout << "Device: " << deviceName << endl;

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    checkError(err, "Creating context");

    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, nullptr, &err);
    checkError(err, "Creating command queue");

    cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(double) * n * n, flatA.data(), &err);
    checkError(err, "Creating buffer A");

    cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(double) * n * n, flatB.data(), &err);
    checkError(err, "Creating buffer B");

    cl_mem bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                    sizeof(double) * n * n, nullptr, &err);
    checkError(err, "Creating buffer C");

    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, nullptr, &err);
    checkError(err, "Creating program");

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        vector<char> log(logSize);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
        cerr << "Build error: " << log.data() << endl;
        checkError(err, "Building program");
    }

    cl_kernel kernel = clCreateKernel(program, "matrixMul", &err);
    checkError(err, "Creating kernel");

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &n);
    checkError(err, "Setting kernel arguments");

    size_t maxWorkGroupSize;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWorkGroupSize), &maxWorkGroupSize, nullptr);
    
    size_t localSize = 16;
    if (maxWorkGroupSize >= 256) localSize = 16;
    if (maxWorkGroupSize >= 1024) localSize = 32;
    
    size_t globalSize = ((n + localSize - 1) / localSize) * localSize;
    size_t localSize2D[2] = {localSize, localSize};
    size_t globalSize2D[2] = {globalSize, globalSize};

    auto start = high_resolution_clock::now();

    err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalSize2D, localSize2D, 0, nullptr, nullptr);
    checkError(err, "Enqueuing kernel");

    clFinish(queue);

    auto end = high_resolution_clock::now();
    double elapsed_ms = duration<double, milli>(end - start).count();

    err = clEnqueueReadBuffer(queue, bufferC, CL_TRUE, 0,
                              sizeof(double) * n * n, flatC.data(), 0, nullptr, nullptr);
    checkError(err, "Reading buffer C");

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            C[i][j] = flatC[i * n + j];

    writeMatrix(fileC, C);

    long long ops = 2LL * n * n * n;
    double ops_per_sec = (ops / elapsed_ms) * 1000.0;

    cout << fixed << setprecision(3);
    cout << "Matrix size: " << n << " x " << n << endl;
    cout << "Execution time: " << elapsed_ms << " ms" << endl;
    cout << "Operations: " << ops << endl;
    cout << "Performance: " << ops_per_sec / 1e6 << " M ops/sec" << endl;

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}