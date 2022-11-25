#pragma once

#include "matrix.h"
#include <omp.h>

bool safe_check(Matrix *src1, Matrix *src2, Matrix *dst);

bool matmul_plain(Matrix *src1, Matrix *src2, Matrix *dst);

bool multiply_strassen(Matrix *src1, Matrix *src2, Matrix *dst);

bool matmul_improved(Matrix *src1, Matrix *src2, Matrix *dst);
