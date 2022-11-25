#include <stdio.h>
#include <stdlib.h>
#include "cblas.h"

int main()
{
    srand(time(0));
    int nn=100;
    int n=nn*nn;
    float a[3] = { 2, 3, 4 };
	float b[3] = { 1, 0, 1 };
	float c[1] = { 0 };
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 1, 1, 3, 1.0, a, 3, b, 3, 0.0, c, 1);
    printf("%f\n",c[0]);
    float * A=(float *)malloc(sizeof(float)*n);
    float * B=(float *)malloc(sizeof(float)*n);
    float * C=(float *)malloc(sizeof(float)*n);
    for(int i=0;i<n;i++)
    {
        A[i]=0.5;
        B[i]=0.5;
    }
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,nn,nn,nn,1.0,A,nn,B,nn,0.0,C,1);
    // int n;      /*! array size */
    // double da;  /*! double constant */
    // double *dx; /*! input double array */
    // int incx;   /*! input stride */
    // double *dy; /*! output double array */
    // int incy;   /*! output stride */

    // int i;

    // n = 10;
    // da = 10;
    // dx = (double *)malloc(sizeof(double) * n);
    // incx = 1;
    // dy = (double *)malloc(sizeof(double) * n);
    // incy = 1;

    // for (i = 0; i < n; i++)
    // {
    //     dx[i] = 9 - i;
    //     dy[i] = i;
    //     printf("%f ", dy[i]); //输出原来的dy
    // }
    // printf("\n");

    // cblas_daxpy(n, da, dx, incx, dy, incy); //运行daxpy程序
    //                                         //    cblas_dcopy(n, dx,incx, dy, incy);      //运行dcopy程序

    // for (i = 0; i < n; i++)
    // {
    //     printf("%f ", dy[i]); //输出计算后的dy
    // }
    // printf("\n");

    // return 0;
}