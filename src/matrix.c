#include "matrix.h"

Matrix *createMat(size_t row, size_t col)
{
    if (row==0||col==0)
    {
        fprintf(stderr, "Rows/cols number is 0.\n");
        return NULL;
    }
    Matrix *pMatrix = malloc(sizeof(Matrix));
    if (pMatrix == NULL)
    {
        fprintf(stderr, "Failed to allocate memory for a matrix.\n");
        return NULL;
    }
    pMatrix->row = row;
    pMatrix->col = col;
    pMatrix->data = malloc(sizeof(float) * row * col);
    if (pMatrix->data == NULL)
    {
        fprintf(stderr, "Failed to allocate memory for the matrix data.\n");
        free(pMatrix);
        return NULL;
    }
    memset(pMatrix->data,0,sizeof(float)*row*col);
    return pMatrix;
}

Matrix *createMatFromArr(size_t row, size_t col,float *src)
{
    if (row==0||col==0)
    {
        fprintf(stderr, "Rows/cols number is 0.\n");
        return NULL;
    }
    if(src==NULL)
    {
        fprintf(stderr,"Source array is NULL.\n");
        return NULL;
    }
    Matrix *pMatrix = malloc(sizeof(Matrix));
    if (pMatrix == NULL)
    {
        fprintf(stderr, "Failed to allocate memory for a matrix.\n");
        return NULL;
    }
    pMatrix->row = row;
    pMatrix->col = col;
    pMatrix->data = malloc(sizeof(float) * row * col);
    if (pMatrix->data == NULL)
    {
        fprintf(stderr, "Failed to allocate memory for the matrix data.\n");
        free(pMatrix);
        return NULL;
    }
    memcpy(pMatrix->data,src,sizeof(float)*row*col);
    return pMatrix;
}

bool releaseMat(Matrix **pMatrix)
{
    if (pMatrix == NULL)
    {
        fprintf(stderr,"Pointer to the pointer of matrix is NULL.\n");
        return false;
    }
    if((*pMatrix)==NULL)
    {
        fprintf(stderr,"The pointer to the matrix is NULL.\n");
        return false;
    }
    if((*pMatrix)->data==NULL)
    {
        fprintf(stderr,"The pointer to the matrix data is NULL.\n");
        return false;
    }
    free((*pMatrix)->data);
    free(*pMatrix);
    *pMatrix = NULL;
    return true;
}

bool equals(Matrix *src1, Matrix *src2)
{
    if(src1==NULL||src2==NULL)
    {
        fprintf(stderr,"The pointer of source matrix is NULL.\n");
        return false;
    }
    if(src1->data==NULL||src2->data==NULL)
    {
        fprintf(stderr,"The pointer to the matrix data is NULL.\n");
        return false;
    }
    if(src1->row==src2->row&&src1->col==src2->col)
    {
        size_t siz=src1->row*src1->col;
        for(size_t i=0;i<siz;i++)
        {
            if(!float_equal(src1->data[i],src2->data[i]))
            {
                printf("difference found: %f : %f\n",src1->data[i],src2->data[i]);
                return false;
            }
        }
        return true;
    }
    return false;
}