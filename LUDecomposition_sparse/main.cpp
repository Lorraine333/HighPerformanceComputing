/*
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * This sample implements a preconditioned conjugate gradient solver on
 * the GPU using CUBLAS and CUSPARSE.  Relative to the conjugateGradient
 * SDK example, this demonstrates the use of cusparseScsrilu0() for
 * computing the incompute-LU preconditioner and cusparseScsrsv_solve()
 * for solving triangular systems.  Specifically, the preconditioned
 * conjugate gradient method with an incomplete LU preconditioner is
 * used to solve the Laplacian operator in 2D on a uniform mesh.
 *
 * Note that the code in this example and the specific matrices used here
 * were chosen to demonstrate the use of the CUSPARSE library as simply
 * and as clearly as possible.  This is not optimized code and the input
 * matrices have been chosen for simplicity rather than performance.
 * These should not be used either as a performance guide or for
 * benchmarking purposes.
 *
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// CUDA Runtime
#include <cuda_runtime.h>

// Using updated (v2) interfaces for CUBLAS and CUSPARSE
#include <cusparse_v2.h>
#include <cublas_v2.h>

// Utilities and system includes
#include <helper_functions.h>  // helper for shared functions common to CUDA SDK samples
#include <helper_cuda.h>       // helper for CUDA error checking

const char *sSDKname     = "conjugateGradientPrecond";

FILE   *fd1; 
FILE   *fd2;
FILE   *fd3;

/* Solve Ax=b using the conjugate gradient method a) without any preconditioning, b) using an Incomplete Cholesky preconditioner and c) using an ILU0 preconditioner. */
int main(int argc, char **argv)
{
    const int max_iter = 1000;
    int k, M = 0, N = 0, nz = 0, *I = NULL, *J = NULL;
    int *d_col, *d_row;
    int qatest = 0;
    const float tol = 1e-12f;
    float *x, *rhs;
	float *lu;
    float r0, r1, alpha, beta;
    float *d_val, *d_x;
    float *d_zm1, *d_zm2, *d_rm2;
    float *d_r, *d_p, *d_omega, *d_y;
    float *val = NULL;
    float *d_valsILU0;
    float *valsILU0;
    float rsum, diff, err = 0.0;
    float qaerr1, qaerr2 = 0.0;
    float dot, numerator, denominator, nalpha;
	const float floatone = 1.0;
	const float floatzero = 0.0;
	int n1=0,n2=0,n3=0,n4=0;
	int temp1,temp2;
	float temp3;
	float *temp;
	float *y;

    int nErrors = 0;

    printf("conjugateGradientPrecond starting...\n");

    /* QA testing mode */
    if (checkCmdLineFlag(argc, (const char **)argv, "qatest"))
    {
        qatest = 1;
    }

    /* This will pick the best possible CUDA capable device */
    cudaDeviceProp deviceProp;
    int devID = findCudaDevice(argc, (const char **)argv);
    printf("GPU selected Device ID = %d \n", devID);

    if (devID < 0)
    {
        printf("Invalid GPU device %d selected,  exiting...\n", devID);
        exit(EXIT_SUCCESS);
    }

    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

    /* Statistics about the GPU device */
    printf("> GPU device has %d Multi-Processors, SM %d.%d compute capabilities\n\n",
           deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

    int version = (deviceProp.major * 0x10 + deviceProp.minor);

    if (version < 0x11)
    {
        printf("%s: requires a minimum CUDA compute 1.1 capability\n", sSDKname);
        cudaDeviceReset();
        exit(EXIT_SUCCESS);
    }


	M = N = 102;
	nz=432;
    I = (int *)malloc(sizeof(int)*(N+1));                              // csr row pointers for matrix A
    J = (int *)malloc(sizeof(int)*nz);                                 // csr column indices for matrix A
    val = (float *)malloc(sizeof(float)*nz);                           // csr values for matrix A
    x = (float *)malloc(sizeof(float)*N);
    rhs = (float *)malloc(sizeof(float)*N);
	lu = (float *)malloc(sizeof(float)*nz);
	y = (float *)malloc(sizeof(float)*N);
	temp = (float *)malloc(sizeof(float)*N);

    for (int i1 = 0; i1 < N; i1++)
    {
        rhs[i1] = 1.0;                                              
        x[i1] = 0.0;      
		y[i1]=0.0;
		temp[i1]=0.0;
    }
	for (int i2 = 0; i2 < nz; i2++)
	{
		lu[i2]=0.0;                                                
	}

		//open the file

	fd1=fopen("Matrix_I_sparse.txt","r");
	if(!fd1)
		printf("fail to open!\n");
	else
	{
		while(n1<N+1)
		{
			fscanf(fd1,"%d",&temp1); 
			I[n1]=temp1;
			n1++;
		}
	}
	fclose(fd1);


	//open file
	fd2=fopen("Matrix_J_sparse.txt","r");
	if(!fd2)
		printf("fail to open!\n");
	else
	{
		while(n2<nz)
		{
			fscanf(fd2,"%d",&temp2); 
			J[n2]=temp2;
			n2++;
		}
	}
	fclose(fd2);

	//open file
	fd3=fopen("Matrix_val_sparse.txt","r");
	if(!fd3)
		printf("fail to open!\n");
	else
	{
		while(n3<nz)
		{
			fscanf(fd3,"%f",&temp3); 
			val[n3]=temp3;
			n3++;
		}
	}
	fclose(fd3);


	/*I[0]=0;I[1]=2;I[2]=5;I[3]=7;
	J[0]=0;J[1]=1;J[2]=0;J[3]=1;J[4]=2;J[5]=1;J[6]=2;
	val[0]=4;val[1]=3;val[2]=3;val[3]=4;val[4]=-1;val[5]=-1;val[6]=4;
	rhs[0]=24;rhs[1]=30;rhs[2]=-24;*/


    //genLaplace(I, J, val, M, N, nz, rhs);

    /* Create CUBLAS context */
    cublasHandle_t cublasHandle = 0;
    cublasStatus_t cublasStatus;
    cublasStatus = cublasCreate(&cublasHandle);

    checkCudaErrors(cublasStatus);

    /* Create CUSPARSE context */
    cusparseHandle_t cusparseHandle = 0;
    cusparseStatus_t cusparseStatus;
    cusparseStatus = cusparseCreate(&cusparseHandle);

    checkCudaErrors(cusparseStatus);

    /* Description of the A matrix*/
    cusparseMatDescr_t descr = 0;
    cusparseStatus = cusparseCreateMatDescr(&descr);

    checkCudaErrors(cusparseStatus);

    /* Define the properties of the matrix */
    cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);

    /* Allocate required memory */
    checkCudaErrors(cudaMalloc((void **)&d_col, nz*sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_row, (N+1)*sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_val, nz*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_x, N*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_y, N*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_r, N*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_p, N*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_omega, N*sizeof(float)));

    cudaMemcpy(d_col, J, nz*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row, I, (N+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val, val, nz*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_r, rhs, N*sizeof(float), cudaMemcpyHostToDevice);
    
	/*LU decomposition*/

    printf("LU decomposition: \n");

    int nzILU0 = 2*N-1;
    valsILU0 = (float *) malloc(nz*sizeof(float));

    checkCudaErrors(cudaMalloc((void **)&d_valsILU0, nz*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_zm1, (N)*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_zm2, (N)*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_rm2, (N)*sizeof(float)));

    /* create the analysis info object for the A matrix */
    cusparseSolveAnalysisInfo_t infoA = 0;
    cusparseStatus = cusparseCreateSolveAnalysisInfo(&infoA);

    checkCudaErrors(cusparseStatus);

    /* Perform the analysis for the Non-Transpose case */
    cusparseStatus = cusparseScsrsv_analysis(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                             N, nz, descr, d_val, d_row, d_col, infoA);

    checkCudaErrors(cusparseStatus);

    /* Copy A data to ILU0 vals as input*/
    cudaMemcpy(d_valsILU0, d_val, nz*sizeof(float), cudaMemcpyDeviceToDevice);
		


    /* generate the Incomplete LU factor H for the matrix A using cudsparseScsrilu0 */
	cusparseStatus = cusparseScsrilu0(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, descr, d_valsILU0, d_row, d_col, infoA);

	checkCudaErrors(cusparseStatus);

	cudaMemcpy(lu, d_valsILU0, nz*sizeof(float), cudaMemcpyDeviceToHost);
	for(int a1=0;a1<nz;a1++)
	{
		printf("d_lu=%f,number=%d\n",lu[a1],a1);
	}

	int i3=0;
	y[i3]=rhs[i3];
	int n=0;

	/*calculate y*/
	i3++;
	
	while(i3<N)
	{
		/*printf("remainder=%d\n",i3%3);*/
		if(i3<60)
		{
			switch(i3%3)
			{ 
			case 0:
				temp[i3]=0.0;                  
				break;
			case 1:
				temp[i3]=lu[3*i3]*y[i3-1];
				break;
			case 2:
				temp[i3]=lu[3*i3]*y[i3-2]+lu[3*i3+1]*y[i3-1];
				break;
			default:
				temp[i3]=0.0;
				break;
			}		
		}

		else
		{
			switch((i3-60)%6)
			{ 
			case 0:
				temp[i3]=0.0;                 
				break;
			case 1:
				temp[i3]=lu[I[i3]]*y[i3-1];
				break;
			case 2:
				temp[i3]=lu[I[i3]]*y[i3-1]+lu[I[i3]+1]*y[i3-2];
				break;
			case 3:
				temp[i3]=lu[I[i3]]*y[i3-1]+lu[I[i3]+1]*y[i3-2]+lu[I[i3]+2]*y[i3-3]; 
				break;
			case 4:
				temp[i3]=lu[I[i3]]*y[i3-1]+lu[I[i3]+1]*y[i3-2]+lu[I[i3]+2]*y[i3-3]+lu[I[i3]+3]*y[i3-4];
				break;
			case 5:
				temp[i3]=lu[I[i3]]*y[i3-1]+lu[I[i3]+1]*y[i3-2]+lu[I[i3]+2]*y[i3-3]+lu[I[i3]+3]*y[i3-4]+lu[I[i3]+4]*y[i3-5];
				break;
			default:
				temp[i3]=0.0;
				break;
			}
		}
	/*	printf("temp=%f\n",temp[i3]);*/
		y[i3]=rhs[i3]-temp[i3];	
		i3++;
	}

	//for(int a2=0;a2<N;a2++)
	//{
	//	printf("y=%f,number=%d\n",y[a2],a2);
	//}

	/*calculate x*/
	for(int a1=0;a1<N;a1++)
	{
		temp[a1]=0;
	}

	int j=N-1;//101
	x[j]=y[j]/lu[nz-1];

	while(j>0)
	{
		j--;
		if(j>=60)
		{
			switch(j%6)
			{ 
			case 0:
				temp[j]=lu[I[j+1]-5]*x[j+1]+lu[I[j+1]-4]*x[j+2]+lu[I[j+1]-3]*x[j+3]+lu[I[j+1]-2]*x[j+4]+lu[I[j+1]-1]*x[j+5];                 
				break;
			case 1:
				temp[j]=lu[I[j+1]-4]*x[j+1]+lu[I[j+1]-3]*x[j+2]+lu[I[j+1]-2]*x[j+3]+lu[I[j+1]-1]*x[j+4];
				break;
			case 2:
				temp[j]=lu[I[j+1]-3]*x[j+1]+lu[I[j+1]-2]*x[j+2]+lu[I[j+1]-1]*x[j+3];
				break;
			case 3:
				temp[j]=lu[I[j+1]-2]*x[j+1]+lu[I[j+1]-1]*x[j+2];
				break;
			case 4:
				temp[j]=lu[I[j+1]-1]*x[j+1];
				break;
			case 5:
				temp[j]=0.0; 
				break;
			default:
				temp[j]=0.0;
				break;
			}	
			x[j]=(y[j]-temp[j])/lu[I[j+1]-5+(j%6)-1];
		}
		else
		{
			switch(j%3)
			{ 
			case 0:
				temp[j]=lu[I[j+1]-2]*x[j+1]+lu[I[j+1]-1]*x[j+2];         
				break;
			case 1:
				temp[j]=lu[I[j+1]-1]*x[j+1];
				break;
			case 2:
				temp[j]=0.0; 
				break;
			default:
				
				break;
			}	
			x[j]=(y[j]-temp[j])/lu[I[j+1]-2+(j%3)-1];
		}
		
	}

	for(int a2=0;a2<N;a2++)
	{
		printf("d_x=%f,number=%d\n",x[a2],a2);
	}

    /* Destroy paramters */
    cusparseDestroySolveAnalysisInfo(infoA);
   // cusparseDestroySolveAnalysisInfo(info_u);

    /* Destroy contexts */
    cusparseDestroy(cusparseHandle);
    cublasDestroy(cublasHandle);

    /* Free device memory */
    free(I);
    free(J);
    free(val);
    free(x);
    free(rhs);
    free(valsILU0);
    cudaFree(d_col);
    cudaFree(d_row);
    cudaFree(d_val);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_r);
    cudaFree(d_p);
    cudaFree(d_omega);
    cudaFree(d_valsILU0);
    cudaFree(d_zm1);
    cudaFree(d_zm2);
    cudaFree(d_rm2);

    cudaDeviceReset();

    printf("  Test Summary:\n");
    printf("     Counted total of %d errors\n", nErrors);
    printf("      qaerr2 = %f\n\n",  fabs(qaerr2));
    //exit((nErrors == 0 &&fabs(qaerr1)<1e-5 && fabs(qaerr2) < 1e-5 ? EXIT_SUCCESS : EXIT_FAILURE));
}

