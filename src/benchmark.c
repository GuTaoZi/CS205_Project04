#include "matmul.h"
#include "matrix.h"
#include <cblas.h>
#include <stdio.h>
#include <time.h>

int main()
{
    clock_t start, end;
    size_t n;
    scanf("%d",&n);
    size_t nn = n * n;
    float *t = malloc(sizeof(float) * nn);
    for (int i = 0; i < nn; i++)
    {
        t[i] = 1.0f * rand() / RAND_MAX;
    }

    Matrix *A = createMatFromArr(n, n, t);
    Matrix *B = createMatFromArr(n, n, t);
    Matrix *C = createMat(n, n);
    matmul_plain(A, B, C);
    memset(C->data, 0, sizeof(float) * nn);
    matmul_plain(A, B, C);
    memset(C->data, 0, sizeof(float) * nn);
    double time1 = omp_get_wtime();
    matmul_plain(A, B, C);
    double time2 = omp_get_wtime();
    printf("[Plain] %ld ms used\n", (long int)(1000 * (time2 - time1)));

    memset(C->data, 0, sizeof(float) * nn);
    matmul_improved(A, B, C);
    memset(C->data, 0, sizeof(float) * nn);
    matmul_improved(A, B, C);
    memset(C->data, 0, sizeof(float) * nn);
    time1 = omp_get_wtime();
    matmul_improved(A, B, C);
    time2 = omp_get_wtime();
    printf("[OpenMP] %ld ms used\n", (long int)(1000 * (time2 - time1)));

    float *res = malloc(sizeof(float) * nn);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, t, n, t, n, 0.0, res, n);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, t, n, t, n, 0.0, res, n);
    memset(res, 0, sizeof(float) * nn);
    time1 = omp_get_wtime();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, t, n, t, n, 0.0, res, n);
    time2 = omp_get_wtime();
    printf("[OpenBLAS] %ld ms used\n", (long int)(1000 * (time2 - time1)));
    Matrix *D = createMatFromArr(n, n, res);
    printf(equals(C, D) ? "Result Accepted.\n" : "Wrong Result.\n");
}