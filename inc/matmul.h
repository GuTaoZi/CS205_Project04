#pragma once

#include "matrix.h"
#include <omp.h>

bool safe_check(Matrix *src1, Matrix *src2, Matrix *dst);

bool matmul_plain(Matrix *src1, Matrix *src2, Matrix *dst);

bool matmul_divide(Matrix *src1, Matrix *src2, Matrix *dst);

bool matmul_sse(Matrix *src1, Matrix *src2, Matrix *dst);

bool matmul_strassen(Matrix *src1, Matrix *src2, Matrix *dst);

bool matmul_avx(Matrix *src1, Matrix *src2, Matrix *dst);

bool matmul_omp(Matrix *src1, Matrix *src2, Matrix *dst);
