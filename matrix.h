#pragma once

#include <stdlib.h>
#include <stdbool.h>
#define float_equal(x, y) ((x-y)<1e-5&&(y-x)<1e-5)

typedef struct Matrix_
{
    size_t row;
    size_t col;
    float *data;
} Matrix;

Matrix *createMat(size_t row, size_t col);

bool releaseMat(Matrix **pMatrix);

bool equals(Matrix *src1, Matrix *src2);
