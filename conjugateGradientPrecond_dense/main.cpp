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

/* Solve Ax=b using the conjugate gradient method a) without any preconditioning, b) using an Incomplete Cholesky preconditioner */
int main(int argc, char **argv)
{
	const int max_iter = 1000;
	int k, M = 0, N = 0, nz = 0, *I = NULL, *J = NULL;
	int *d_col, *d_row;
	int qatest = 0;
	const float tol = 1e-12f;
	float *x, *rhs;
	float r0, r1, alpha, beta;
	float *d_val, *d_x;
	float *d_zm1, *d_zm2, *d_rm2;
	float *d_r, *d_p, *d_omega, *d_y;
	float *val = NULL;
	float *d_valsILU0,*d_valsIChol0;
	float *valsILU0,*valsIChol0;
	float rsum, diff, err = 0.0;
	float qaerr1, qaerr2 = 0.0;
	float dot, numerator, denominator, nalpha;
	const float floatone = 1.0;
	const float mfloatone = -1.0;
	const float floatzero = 0.0;
	int n1=0,n2=0,n3=0,n4=0;
	int temp1,temp2;
	float temp3,temp4;
	float *out1=NULL;

	int *UI = NULL, *UJ = NULL;
	float *Uval = NULL;
	int *d_col1, *d_row1;
	float *d_val1;


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


	M = N = 504;
	nz=1512;
	I = (int *)malloc(sizeof(int)*(N+1));                              // csr row pointers for matrix A
	J = (int *)malloc(sizeof(int)*nz);                                 // csr column indices for matrix A
	val = (float *)malloc(sizeof(float)*nz);                           // csr values for matrix A
	x = (float *)malloc(sizeof(float)*N);
	rhs = (float *)malloc(sizeof(float)*N);
	out1 = (float *)malloc(sizeof(float)*nz);


	for (int i = 0; i < N; i++)
	{
		rhs[i] = 0.0;                                                  // Initialize RHS
		x[i] = 0.0;                                                    // Initial approximation of solution
	}

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


	for (int i = 0; i < N; i++)
	{
		rhs[i] = 1.0;
		x[i] = 0.0;
	}
	//I[0]=0;I[1]=2;I[2]=5;I[3]=7;
	//J[0]=0;J[1]=1;J[2]=0;J[3]=1;J[4]=2;J[5]=1;J[6]=2;
	//val[0]=4;val[1]=3;val[2]=3;val[3]=4;val[4]=-1;val[5]=-1;val[6]=4;
	//rhs[0]=24;rhs[1]=30;rhs[2]=-24;


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


	
	
	
	
	///* Conjugate gradient without preconditioning.*/

	//   printf("Convergence of conjugate gradient without preconditioning: \n");
	//   k = 0;
	//   r0 = 0;
	//   cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);	//上面函数主要实现dr与dr的点乘,赋值给r1 

	//   while (r1 > tol*tol && k <= max_iter)
	//   {
	//       k++;

	//       if (k == 1)
	//       {
	//           cublasScopy(cublasHandle, N, d_r, 1, d_p, 1);			//d_p=d_r;
	//       }
	//       else
	//       {
	//           beta = r1/r0;
	//           cublasSscal(cublasHandle, N, &beta, d_p, 1);			//d_p=d_p*beta
	//           cublasSaxpy(cublasHandle, N, &floatone, d_r, 1, d_p, 1);			//完成的计算是d_p=d_p+floatone*d_r
	//       }

	//       cusparseScsrmv(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &floatone, descr, d_val, d_row, d_col, d_p, &floatzero, d_omega);		//上述函数输出结果为0。实际计算为d_omega=floatone*A*d_p+floatzero*d_omega;d_omega=A*d_p;
	//       cublasSdot(cublasHandle, N, d_p, 1, d_omega, 1, &dot);		//上面函数主要实现d_p与d_omega的点乘,赋值给dot
	//       alpha = r1/dot;
	//       cublasSaxpy(cublasHandle, N, &alpha, d_p, 1, d_x, 1);		//完成的计算是d_x=d_x+alpha*d_p
	//       nalpha = -alpha;
	//       cublasSaxpy(cublasHandle, N, &nalpha, d_omega, 1, d_r, 1);		//完成的计算是d_r=d_r-alpha*d_omega
	//       r0 = r1;
	//       cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);		//上面函数主要实现dr与dr的点乘,赋值给r1
	//   }

	//   printf("  iteration = %3d, residual = %e \n", k, sqrt(r1));

	//   cudaMemcpy(x, d_x, N*sizeof(float), cudaMemcpyDeviceToHost);
	//
	///*for(int a1=0;a1<N;a1++)
	//{
	//		printf("d_Ax=%f\n",x[a1]);
	//}*/
	//   /* check result */
	//   err = 0.0;

	//   for (int i = 0; i < N; i++)
	//   {
	//       rsum = 0.0;

	//       for (int j = I[i]; j < I[i+1]; j++)
	//       {
	//           rsum += val[j]*x[J[j]];
	//       }

	//       diff = fabs(rsum - rhs[i]);

	//       if (diff > err)
	//       {
	//           err = diff;
	//       }
	//   }

	//   printf("  Convergence Test: %s \n", (k <= max_iter) ? "OK" : "FAIL");
	//   nErrors += (k > max_iter) ? 1 : 0;
	//   qaerr1 = err;

	//   if (0)
	//   {
	//       // output result in matlab-style array
	//       int n=(int)sqrt((double)N);
	//       printf("a = [  ");

	//       for (int iy=0; iy<n; iy++)
	//       {
	//           for (int ix=0; ix<n; ix++)
	//           {
	//               printf(" %f ", x[iy*n+ix]);
	//           }

	//           if (iy == n-1)
	//           {
	//               printf(" ]");
	//           }

	//           printf("\n");
	//       }
	//   }












	///* Preconditioned Conjugate Gradient using IChol.
	//--------------------------------------------
	//Follows the description by Golub & Van Loan, "Matrix Computations 3rd ed.", Algorithm 10.3.1  */

	printf("\nConvergence of conjugate gradient using incomplete Cholskey preconditioning: \n");
	int nz1=1008;
	int nzIChol0 = 1008;
	
	UI = (int *)malloc(sizeof(int)*(N+1));                              // csr row pointers for matrix A
	UJ = (int *)malloc(sizeof(int)*nz1);                                 // csr column indices for matrix A
	Uval = (float *)malloc(sizeof(float)*nz1);                           // csr values for matrix A
	valsIChol0 = (float *) malloc(nz1*sizeof(float));

	n1=0;
	n2=0;
	n3=0;

	fd1=fopen("Matrix_UI_sparse.txt","r");
	if(!fd1)
		printf("fail to open!\n");
	else
	{
		while(n1<N+1)
		{
			fscanf(fd1,"%d",&temp1); 
			//printf("%d\n",temp1);
			UI[n1]=temp1;
			n1++;
		}
	}
	fclose(fd1);



	fd2=fopen("Matrix_UJ_sparse.txt","r");
	if(!fd2)
		printf("fail to open!\n");
	else
	{
		while(n2<nz1)
		{
			fscanf(fd2,"%d",&temp2); 
			//printf("%d\n",temp2);
			UJ[n2]=temp2;
			n2++;
		}
	}
	fclose(fd2);


	fd3=fopen("Matrix_Uval_sparse.txt","r");
	if(!fd3)
		printf("fail to open!\n");
	else
	{
		while(n3<nz1)
		{
			fscanf(fd3,"%f",&temp3); 
			//printf("%f\n%d",temp3,n3);
			Uval[n3]=temp3;
			n3++;
		}
	}
	fclose(fd3);

	//UI[0]=0;UI[1]=2;UI[2]=4;UI[3]=5;
	//UJ[0]=0;UJ[1]=1;UJ[2]=1;UJ[3]=2;UJ[4]=2;
	//Uval[0]=4;Uval[1]=3;Uval[2]=4;Uval[3]=-1;Uval[4]=2;



	checkCudaErrors(cudaMalloc((void **)&d_valsIChol0, nz1*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_zm1, (N)*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_zm2, (N)*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_rm2, (N)*sizeof(float)));

	/* Description of the upper A matrix*/
	cusparseMatDescr_t descru = 0;
	cusparseStatus = cusparseCreateMatDescr(&descru);

	checkCudaErrors(cusparseStatus);

	/* Define the properties of the matrix */
	cusparseSetMatType(descru,CUSPARSE_MATRIX_TYPE_SYMMETRIC);
	cusparseSetMatIndexBase(descru,CUSPARSE_INDEX_BASE_ZERO);
	cusparseSetMatFillMode(descru,CUSPARSE_FILL_MODE_UPPER);
	cusparseSetMatDiagType(descru, CUSPARSE_DIAG_TYPE_NON_UNIT);

	/* Allocate required memory */
	checkCudaErrors(cudaMalloc((void **)&d_col1, nz1*sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&d_row1, (N+1)*sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&d_val1, nz1*sizeof(float)));

	cudaMemcpy(d_col1, UJ, nz1*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_row1, UI, (N+1)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_val1, Uval, nz1*sizeof(float), cudaMemcpyHostToDevice);


	/* create the analysis info object for the A matrix */
	cusparseSolveAnalysisInfo_t infoA = 0;
	cusparseStatus = cusparseCreateSolveAnalysisInfo(&infoA);

	checkCudaErrors(cusparseStatus);

	/* Perform the analysis for the upper case */
	cusparseStatus = cusparseScsrsv_analysis(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,N, nz1, descru, d_val1, d_row1, d_col1, infoA);
	checkCudaErrors(cusparseStatus);


	/* Copy A data to IChol0 vals as input*/
	cudaMemcpy(d_valsIChol0, d_val1, nz1*sizeof(float), cudaMemcpyDeviceToDevice);

	/* generate the Incomplete Cholskey factor H for the matrix A using cudsparseScsric0 */
	cusparseStatus = cusparseScsric0(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, descru, d_valsIChol0, d_row1, d_col1, infoA);
	checkCudaErrors(cusparseStatus);

	/* Create info objects for the Chol0 preconditioner */
	cusparseSolveAnalysisInfo_t info_R;
	cusparseSolveAnalysisInfo_t info_Rt;
	cusparseCreateSolveAnalysisInfo(&info_R);
	cusparseCreateSolveAnalysisInfo(&info_Rt);

	/* Create description objects for the Chol preconditioner upper matrix */
	cusparseMatDescr_t descrR = 0;
	cusparseStatus = cusparseCreateMatDescr(&descrR);
	cusparseSetMatType(descrR,CUSPARSE_MATRIX_TYPE_TRIANGULAR);
	cusparseSetMatIndexBase(descrR,CUSPARSE_INDEX_BASE_ZERO);
	cusparseSetMatFillMode(descrR, CUSPARSE_FILL_MODE_UPPER);
	cusparseSetMatDiagType(descrR, CUSPARSE_DIAG_TYPE_NON_UNIT);
	cusparseStatus = cusparseScsrsv_analysis(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, nz1, descrR, d_valsIChol0, d_row1, d_col1, info_R);
	cusparseStatus = cusparseScsrsv_analysis(cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE, N, nz1, descrR, d_valsIChol0, d_row1, d_col1, info_Rt);

	/* reset the initial guess of the solution to zero */
	for (int i = 0; i < N; i++)
	{
		rhs[i] = 1.0;
		x[i] = 0.0;
	}

	cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_r, rhs, N*sizeof(float), cudaMemcpyHostToDevice);
	k = 0;
	cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);

	while (r1 > tol*tol && k <= max_iter)
	{
		// Forward Solve, we can re-use infoA since the sparsity pattern of A matches that of L
		cusparseStatus = cusparseScsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE, N, &floatone,  descrR, d_valsIChol0, d_row1, d_col1, info_Rt, d_r, d_y);
		checkCudaErrors(cusparseStatus);
		cusparseStatus = cusparseScsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, &floatone,  descrR, d_valsIChol0, d_row1, d_col1, info_R, d_y, d_zm1);
		checkCudaErrors(cusparseStatus);



		k++;

		if (k == 1)
		{
			cublasScopy(cublasHandle, N, d_zm1, 1, d_p, 1);
			//d_p=d_zm1;
		}
		else
		{
			cublasSdot(cublasHandle, N, d_r, 1, d_zm1, 1, &numerator);//numerator=d_r*d_zm1
			cublasSdot(cublasHandle, N, d_rm2, 1, d_zm2, 1, &denominator);//denominator=d_rm2*d_zm2
			beta = numerator/denominator;
			cublasSscal(cublasHandle, N, &beta, d_p, 1);//d_p=d_p*beta
			cublasSaxpy(cublasHandle, N, &floatone, d_zm1, 1, d_p, 1); //d_p=d_p+floattone*d_zm1;
		}

		cusparseScsrmv(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &floatone, descr, d_val, d_row, d_col, d_p, &floatzero, d_omega);//实际计算为d_omega=floatone*A*d_p+floatzero*d_omega;d_omega=A*d_p;
		cublasSdot(cublasHandle, N, d_r, 1, d_zm1, 1, &numerator);               //numerator=d_r*d_zm1;
		cublasSdot(cublasHandle, N, d_p, 1, d_omega, 1, &denominator);            //donominator=d_p*d_omega;
		alpha = numerator / denominator;
		cublasSaxpy(cublasHandle, N, &alpha, d_p, 1, d_x, 1);                   //d_x=d_x+alpha*d_p;
		cublasScopy(cublasHandle, N, d_r, 1, d_rm2, 1);                 //d_rm2=d_r;
		cublasScopy(cublasHandle, N, d_zm1, 1, d_zm2, 1);                 //d_zm2=d_zm1
		nalpha = -alpha;
		cublasSaxpy(cublasHandle, N, &nalpha, d_omega, 1, d_r, 1);              //d_r=d_r-alpha*d_omega;
		cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);                        //r1=d_r*d_r
	}

	printf("  iteration = %3d, residual = %e \n", k, sqrt(r1));


	cudaMemcpy(x, d_x, N*sizeof(float), cudaMemcpyDeviceToHost);

	//for(int a1=0;a1<N;a1++)
	//{
	//	printf("d_Ax=%f\n",x[a1]);
	//}

	/* check result */
	err = 0.0;

	for (int i = 0; i < N; i++)
	{
		rsum = 0.0;

		for (int j = I[i]; j < I[i+1]; j++)
		{
			rsum += val[j]*x[J[j]];
		}

		diff = fabs(rsum - rhs[i]);

		if (diff > err)
		{
			err = diff;
		}
	}

	printf("  Convergence Test: %s \n", (k <= max_iter) ? "OK" : "FAIL");
	nErrors += (k > max_iter) ? 1 : 0;
	qaerr2 = err;

	/* Destroy paramters */
	cusparseDestroySolveAnalysisInfo(infoA);
	cusparseDestroySolveAnalysisInfo(info_Rt);
	cusparseDestroySolveAnalysisInfo(info_R);

	/* Destroy contexts */
	cusparseDestroy(cusparseHandle);
	cublasDestroy(cublasHandle);

	/* Free device memory */
	free(I);
	free(J);
	free(val);
	free(UI);
	free(UJ);
	free(Uval);
	free(x);
	free(rhs);
	free(valsIChol0);
	cudaFree(d_col);
	cudaFree(d_row);
	cudaFree(d_col1);
	cudaFree(d_row1);
	cudaFree(d_val);
	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_r);
	cudaFree(d_p);
	cudaFree(d_omega);
	cudaFree(d_valsIChol0);
	cudaFree(d_zm1);
	cudaFree(d_zm2);
	cudaFree(d_rm2);

	cudaDeviceReset();

	printf("  Test Summary:\n");
	printf("     Counted total of %d errors\n", nErrors);
//	printf("     qaerr1 = %f qaerr2 = %f\n\n", fabs(qaerr1), fabs(qaerr2));
//	exit((nErrors == 0 &&fabs(qaerr1)<1e-5 && fabs(qaerr2) < 1e-5 ? EXIT_SUCCESS : EXIT_FAILURE));
}

