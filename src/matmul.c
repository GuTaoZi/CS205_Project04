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
    return true;
}

bool matmul_avx_vec8(Matrix *src1, Matrix *src2, Matrix *dst)
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
    return true;
}

bool matmul_omp(Matrix *src1, Matrix *src2, Matrix *dst)
{
    if (!safe_check(src1, src2, dst))
    {
        return false;
    }
    register size_t n = src1->row;
    register size_t k, j;
#pragma omp parallel
    {
#pragma omp for private(k, j)
        for (register size_t i = 0; i < n; i++)
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

bool matmul_avx_block8(Matrix *src1, Matrix *src2, Matrix *dst)
{
    if (!safe_check(src1, src2, dst))
    {
        return false;
    }
    register size_t n = src1->row;
    float *data1 = src1->data;
    float *data2 = src2->data;
    float *data3 = dst->data;

#pragma omp parallel for
    for (register size_t i = 0; i < n; i += 8)
    {
        for (register size_t j = 0; j < n; j += 8)
        {
            for (register size_t k = 0; k < n; k += 8)
            {
                for (register size_t u = i; u < i + 8; u++)
                {
                    __m256 b_1 = _mm256_loadu_ps(data2 + j * n + k);
                    __m256 b_2 = _mm256_loadu_ps(data2 + (j + 1) * n + k);
                    __m256 b_3 = _mm256_loadu_ps(data2 + (j + 2) * n + k);
                    __m256 b_4 = _mm256_loadu_ps(data2 + (j + 3) * n + k);
                    __m256 b_5 = _mm256_loadu_ps(data2 + (j + 4) * n + k);
                    __m256 b_6 = _mm256_loadu_ps(data2 + (j + 5) * n + k);
                    __m256 b_7 = _mm256_loadu_ps(data2 + (j + 6) * n + k);
                    __m256 b_8 = _mm256_loadu_ps(data2 + (j + 7) * n + k);

                    __m256 a_1 = _mm256_set1_ps(data1[u * n + j]);
                    __m256 a_2 = _mm256_set1_ps(data1[u * n + j + 1]);
                    __m256 a_3 = _mm256_set1_ps(data1[u * n + j + 2]);
                    __m256 a_4 = _mm256_set1_ps(data1[u * n + j + 3]);
                    __m256 a_5 = _mm256_set1_ps(data1[u * n + j + 4]);
                    __m256 a_6 = _mm256_set1_ps(data1[u * n + j + 5]);
                    __m256 a_7 = _mm256_set1_ps(data1[u * n + j + 6]);
                    __m256 a_8 = _mm256_set1_ps(data1[u * n + j + 7]);

                    b_1 = _mm256_mul_ps(b_1, a_1);
                    b_2 = _mm256_mul_ps(b_2, a_2);
                    b_3 = _mm256_mul_ps(b_3, a_3);
                    b_4 = _mm256_mul_ps(b_4, a_4);
                    b_5 = _mm256_mul_ps(b_5, a_5);
                    b_6 = _mm256_mul_ps(b_6, a_6);
                    b_7 = _mm256_mul_ps(b_7, a_7);
                    b_8 = _mm256_mul_ps(b_8, a_8);

                    __m256 temp_1 = _mm256_add_ps(b_1, b_2);
                    __m256 temp_2 = _mm256_add_ps(b_3, b_4);
                    __m256 temp_3 = _mm256_add_ps(b_5, b_6);
                    __m256 temp_4 = _mm256_add_ps(b_7, b_8);

                    __m256 temp_5 = _mm256_add_ps(temp_1, temp_2);
                    __m256 temp_6 = _mm256_add_ps(temp_3, temp_4);
                    __m256 temp_7 = _mm256_add_ps(temp_5, temp_6);

                    __m256 temp_c = _mm256_loadu_ps(data3 + u * n + k);
                    __m256 temp_8 = _mm256_add_ps(temp_7, temp_c);

                    _mm256_storeu_ps(data3 + u * n + k, temp_8);
                }
            }
        }
    }
    return true;
}

// #include <pthread.h>

// struct PartialMatMulParams
// {
//     size_t fromColumn, toColum, n;
//     float *a, *b, *c;
// };

// void *partialMatMul(void *params)
// {
//     struct PartialMatMulParams *p = (struct PartialMatMulParams *)params;
//     size_t n = p->n;
//     float *a = p->a;
//     float *b = p->b;
//     float *c = p->c;

// #pragma omp parallel for
//     for (size_t i = p->fromColumn; i < p->toColum; i++)
//     {
//         for (size_t j = 0; j < n; j++)
//         {
//             for (size_t k = 0; k < n; k++)
//             {
//                 c[j * n + i] += a[j * n + k] * b[k * n + i];
//             }
//         }
//     }
//     return NULL;
// }

// bool matmul_thread(Matrix *src1, Matrix *src2, Matrix *dst, size_t num_threads)
// {
//     if (!safe_check(src1, src2, dst))
//     {
//         return false;
//     }
//     register size_t n = src1->row;
//     float *data1 = src1->data;
//     float *data2 = src2->data;
//     float *data3 = dst->data;
//     pthread_t *threads = malloc(sizeof(pthread_t) * num_threads);
//     struct PartialMatMulParams *params = malloc(sizeof(struct PartialMatMulParams) * num_threads);

//     for (size_t i = 0; i < num_threads; i++)
//     {
//         params[i].a = data1;
//         params[i].b = data2;
//         params[i].c = data3;
//         params[i].n = n;

//         params[i].fromColumn = i * (n / num_threads);
//         params[i].toColum = (i + 1) * (n / num_threads);
//         pthread_create(&threads[i], NULL, partialMatMul, &params[i]);
//     }
//     for (size_t i = 0; i < num_threads; i++)
//     {
//         pthread_join(threads[i], NULL);
//     }
//     free(threads);
//     free(params);
// }
