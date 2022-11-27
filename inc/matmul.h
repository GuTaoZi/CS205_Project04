#pragma once

#include "matrix.h"
#include <omp.h>

bool safe_check(Matrix *src1, Matrix *src2, Matrix *dst);

bool matmul_plain(Matrix *src1, Matrix *src2, Matrix *dst);

bool matmul_divide(Matrix *src1, Matrix *src2, Matrix *dst);

bool matmul_omp(Matrix *src1, Matrix *src2, Matrix *dst);

bool matmul_avx_vec8(Matrix *src1, Matrix *src2, Matrix *dst);

bool matmul_avx_block8(Matrix *src1, Matrix *src2, Matrix *dst);

bool matmul_thread(Matrix *src1, Matrix *src2, Matrix *dst, size_t num_threads);
