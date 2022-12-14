#pragma once

#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <memory.h>
#define float_equal(x, y) ((x - y) < 1e-2 && (y - x) < 1e-2)
#define mx(x, y) ((x) > (y) ? (x) : (y))
#define float_equal2(x, y) (x > y ? (x - y < mx(x, 1) * 1e-3) : (y - x < mx(y, 1) * 1e-3))
// #define float_equal(x, y) (x==y)

typedef struct Matrix_
{
    size_t row;
    size_t col;
    float *data;
} Matrix;

Matrix *createMat(size_t row, size_t col);

Matrix *createMatFromArr(size_t row, size_t col, float *src);

bool releaseMat(Matrix **pMatrix);

bool equals(Matrix *src1, Matrix *src2);
