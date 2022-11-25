#include "matmul.h"
#include <stdio.h>

bool safe_check(Matrix *src1, Matrix *src2, Matrix *dst)
{
    if (src1 == NULL)
    {
        fprintf(stderr, "File %s, Line %s, Function %s(): The 1st parameter is NULL.\n", __FILE__, __LINE__,
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
        fprintf(stderr, "File %s, Line %s, Function %s(): The 2nd parameter is NULL.\n", __FILE__, __LINE__,
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
        fprintf(stderr, "File %s, Line %s, Function %s(): The 3rd parameter is NULL.\n", __FILE__, __LINE__,
                __FUNCTION__);
        return false;
    }
    else if (dst->data == NULL)
    {
        fprintf(stderr, "%s(): The 3rd parameter has no valid data.\n", __FUNCTION__);
        return false;
    }
    if (src1->row!=src1->col||src1->row != src2->row || src1->col != src2->col || src1->row != dst->row || src1->col != dst->col)
    {
        fprintf(stderr, "The input and the output do not match, they should have the same square size.\n");
        fprintf(stderr, "Their sizes are (%zu,%zu), (%zu,%zu) and (%zu,%zu).\n",
                src1->row, src1->col, src2->row, src2->col, dst->row, dst->col);
        return false;
    }
}

bool matmul_plain(Matrix *src1, Matrix *src2, Matrix *dst)
{
    if(!safe_check(src1,src2,dst))
    {
        return false;
    }
    size_t n=src1->row;
    for(size_t i=0;i<n;i++)
    {
        size_t i_n=i*n;
        for(size_t k=0;k<n;k++)
        {
            float t=src1->data[i_n+k];
            size_t k_n=k*n;
            for(size_t j=0;j<n;j++)
            {
                dst->data[i_n+j]+=t*src2->data[k_n+j];
            }
        }
    }
    return true;
}

bool multiply_strassen(Matrix *src1, Matrix *src2, Matrix *dst);

bool matmul_improved(Matrix *src1, Matrix *src2, Matrix *dst);