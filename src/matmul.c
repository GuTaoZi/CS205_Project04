#include "matmul.h"
#include <immintrin.h>

bool safe_check(Matrix *src1, Matrix *src2, Matrix *dst)
{
    if (src1 == NULL)
    {
        fprintf(stderr, "File %s, Line %d, Function %s(): The 1st parameter is NULL.\n", __FILE__, __LINE__,
                __FUNCTION__);
        return false;
    }
    else if (src1->data == NULL)
    {
        fprintf(stderr, "%s(): The 1st parameter has no valid data.\n", __FUNCTION__);
        return false;
    }
    if (src2 == NULL)
    {
        fprintf(stderr, "File %s, Line %d, Function %s(): The 2nd parameter is NULL.\n", __FILE__, __LINE__,
                __FUNCTION__);
        return false;
    }
    else if (src2->data == NULL)
    {
        fprintf(stderr, "%s(): The 2nd parameter has no valid data.\n", __FUNCTION__);
        return false;
    }
    if (dst == NULL)
    {
        fprintf(stderr, "File %s, Line %d, Function %s(): The 3rd parameter is NULL.\n", __FILE__, __LINE__,
                __FUNCTION__);
        return false;
    }
    else if (dst->data == NULL)
    {
        fprintf(stderr, "%s(): The 3rd parameter has no valid data.\n", __FUNCTION__);
        return false;
    }
    if (src1->row != src1->col || src1->row != src2->row || src1->col != src2->col || src1->row != dst->row || src1->col != dst->col)
    {
        fprintf(stderr, "The input and the output do not match, they should have the same square size.\n");
        fprintf(stderr, "Their sizes are (%zu,%zu), (%zu,%zu) and (%zu,%zu).\n",
                src1->row, src1->col, src2->row, src2->col, dst->row, dst->col);
        return false;
    }
    return true;
}

float *transpose(float *src, size_t n)
{
    float *res = malloc(sizeof(float) * n * n);
    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            res[j * n + i] = src[i * n + j];
        }
    }
    return res;
}

bool matmul_plain(Matrix *src1, Matrix *src2, Matrix *dst)
{
    if (!safe_check(src1, src2, dst))
    {
        return false;
    }
    size_t n = src1->row;
    for (size_t i = 0; i < n; i++)
    {
        size_t i_n = i * n;
        for (size_t k = 0; k < n; k++)
        {
            float t = src1->data[i_n + k];
            size_t k_n = k * n;
            for (size_t j = 0; j < n; j++)
            {
                dst->data[i_n + j] += t * src2->data[k_n + j];
            }
        }
    }
    return true;
}

bool matmul_divide(Matrix *src1, Matrix *src2, Matrix *dst)
{
    if (!safe_check(src1, src2, dst))
    {
        return false;
    }
    size_t n = src1->row;
    size_t k, j;
    float *data1 = src1->data;
    float *data2 = transpose(src2->data, n);
    float *data3 = dst->data;
#pragma omp parallel
    for (size_t i = 0; i < n; i += 4)
    {
#pragma omp for private(k, j)
        for (j = 0; j < n; j += 4)
        {
            for (k = 0; k < n; k += 4)
            {
                for (size_t i2 = 0; i2 < 4; i2++)
                {
                    for (size_t j2 = 0; j2 < 4; j2++)
                    {
                        for (size_t k2 = 0; k2 < 4; k2++)
                        {
                            data3[(i + i2) * n + (j + j2)] += data1[(i + i2) * n + (k + k2)] * data2[(k + k2) + (j + j2) * n];
                        }
                    }
                }
            }
        }
    }
}

bool matmul_avx(Matrix *src1, Matrix *src2, Matrix *dst)
{
    if (!safe_check(src1, src2, dst))
    {
        return false;
    }
    size_t n = src1->row;
    size_t k, j;
    float *data1 = src1->data;
    float *data2 = transpose(src2->data, n);
    float *data3 = dst->data;
    #pragma omp parallel
    for (size_t i = 0; i < n; i++)
    {
        #pragma omp for private(k, j)
        for (j = 0; j < n; j++)
        {
            __m256 sx = _mm256_setzero_ps();
            for (k = 0; k < n; k += 8)
            {
                sx = _mm256_add_ps(sx, _mm256_mul_ps(_mm256_loadu_ps(data1 + i * n + k), _mm256_loadu_ps(data2 + j * n + k)));
            }
            sx = _mm256_add_ps(sx, _mm256_permute2f128_ps(sx, sx, 1));
            sx = _mm256_hadd_ps(sx, sx);
            data3[i * n + j] = _mm256_cvtss_f32(_mm256_hadd_ps(sx, sx));
        }
    }
}

bool matmul_omp(Matrix *src1, Matrix *src2, Matrix *dst)
{
    if (!safe_check(src1, src2, dst))
    {
        return false;
    }
    size_t n = src1->row;
    size_t k, j;
#pragma omp parallel
    {
#pragma omp for private(k, j)
        for (size_t i = 0; i < n; i++)
        {
            size_t i_n = i * n;
            for (k = 0; k < n; k++)
            {
                float t = src1->data[i_n + k];
                size_t k_n = k * n;
                for (j = 0; j < n; j++)
                {
                    dst->data[i_n + j] += t * src2->data[k_n + j];
                }
            }
        }
    }

    return true;
}