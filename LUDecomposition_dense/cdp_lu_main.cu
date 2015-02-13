#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "helper_string.h"
#include "helper_cuda.h"

#include "cdp_lu.h"
#include "cdp_lu_utils.h"



void memsetup(Parameters &host_params)
{
	FILE   *fd1; 
	int n1=0;
	double temp1;
    srand(host_params.seed);

    // Initialise with the base params, and do any necessary randomisation
    unsigned long long lda_ull = (unsigned long long) host_params.lda;//1024
    host_params.flop_count += lda_ull*lda_ull*lda_ull * 2ull / 3ull;

    host_params.data_size = sizeof(double);//DOUBLE SIZE
    host_params.data_len = host_params.n * host_params.lda;//1024*1024
    host_params.piv_len = MIN(host_params.m, host_params.n);//1024

    size_t len = host_params.data_len * host_params.data_size;//1024*1024*DOUBLE SIZE
    size_t piv_len = host_params.piv_len * sizeof(int);//1024*INT SIZE

    // Allocate memories
    host_params.host_A = (double *) malloc(len);//1024*1024*DOUBLE SIZE
    host_params.host_LU = (double *) malloc(len);//1024*1024*DOUBLE SIZE
    host_params.host_piv = (int *) malloc(piv_len);//1024*INT SIZE

    checkCudaErrors(cudaMalloc((void **)&host_params.device_A, len));
    checkCudaErrors(cudaMalloc((void **)&host_params.device_LU, len));
    checkCudaErrors(cudaMalloc((void **)&host_params.device_piv, piv_len));
    checkCudaErrors(cudaMalloc((void **)&host_params.device_info, sizeof(int)));

    // Initialise source with random (seeded) data
    // srand(params[b].seed);


    double *ptr = host_params.host_A;
	/*
    for (int i=0; i<host_params.data_len; i++)
	{
        ptr[i] = (double) rand() / 32768.0;//set A value
	}*/

	ptr[0]=4;ptr[1]=3;ptr[2]=0;ptr[3]=3;ptr[4]=4;ptr[5]=-1;ptr[6]=0;ptr[7]=-1;ptr[8]=4;

	//ptr[0]=-5;ptr[1]=1;ptr[2]=1;ptr[3]=2;ptr[4]=1;ptr[5]=2;
	//ptr[6]=1;ptr[7]=-5;ptr[8]=-1;ptr[9]=1;ptr[10]=3;ptr[11]=1;
	//ptr[12]=1;ptr[13]=-1;ptr[14]=-5;ptr[15]=-1;ptr[16]=1;ptr[17]=4;
	//ptr[18]=2;ptr[19]=1;ptr[20]=-1;ptr[21]=-5;ptr[22]=1;ptr[23]=1;
	//ptr[24]=1;ptr[25]=3;ptr[26]=1;ptr[27]=1;ptr[28]=-5;ptr[29]=1;
	//ptr[30]=2;ptr[31]=1;ptr[32]=4;ptr[33]=1;ptr[34]=1;ptr[35]=-5;

		/*modified*/
	//open file
	//fd1=fopen("Matrix_val_dense.txt","r");
	//if(!fd1)
	//	printf("fail to open!\n");
	//else
	//{
	//	while(n1<host_params.data_len)
	//	{
	//		fscanf(fd1,"%lf",&temp1); 
	//		ptr[n1]=temp1;
	//		n1++;
	//	}
	//}
	//fclose(fd1);


	/*modified*/
    memset(host_params.host_piv, 0, piv_len);
    host_params.host_info = 0;

    memcpy(host_params.host_LU, host_params.host_A, len);   // copy reference data

    // Now upload it to the GPU. TODO: copy A to LU on the device..
    checkCudaErrors(cudaMemcpy(host_params.device_A, host_params.host_A, len, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(host_params.device_LU, host_params.host_LU, len, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(host_params.device_piv, host_params.host_piv, piv_len, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(host_params.device_info, &host_params.device_info, sizeof(int), cudaMemcpyHostToDevice));
}

void finalize(Parameters &host_params)
{
    free(host_params.host_A);
    free(host_params.host_LU);
    free(host_params.host_piv);

    checkCudaErrors(cudaFree(host_params.device_A));
    checkCudaErrors(cudaFree(host_params.device_LU));
    checkCudaErrors(cudaFree(host_params.device_piv));
    checkCudaErrors(cudaFree(host_params.device_info));
}

// Figure out the command line
void set_defaults(Parameters &host_params, int matrix_size)
{
    // Set default params
    host_params.seed = 4321;
    host_params.m = host_params.n = host_params.lda = matrix_size;
}

// This is the main launch entry point. We have to package up our pointers
// into arrays and then decide if we're launching pthreads or CNP blocks.
void launch(Parameters &host_params)
{
    // Create a device-side copy of the params array...

    Parameters *device_params;
    checkCudaErrors(cudaMalloc((void **)&device_params, sizeof(Parameters)));
	checkCudaErrors(cudaMemcpy(device_params, &host_params, sizeof(Parameters), cudaMemcpyHostToDevice));
    dgetrf_test(&host_params, device_params);
	
	double *out1;
	out1=(double*)malloc(host_params.data_len*sizeof(double));
	//cudaMemcpy(out1,host_params.device_LU , host_params.data_len*sizeof(double), cudaMemcpyDeviceToHost);
	//for(int a1=0;a1<host_params.data_len;a1++)
	//{
	//	printf("d_LU=%f\n",out1[a1]);
	//}
    
	checkCudaErrors(cudaFree(device_params));

}

bool checkresult(Parameters &host_params)
{
    printf("Resolving results... \n");

	FILE   *fd2;
	int n2=0;
	double temp2;
    size_t len = host_params.data_len * host_params.data_size;//1024*1024*DOUBLE SIZE
    size_t piv_len = host_params.piv_len * sizeof(int);//1024*INT SIZE

	
	
    double *device_I;
    double *host_I = (double *)malloc(len);
    // initialize iidentity matrix
    memset(host_I, 0, len);//初始化

    for (int i = 0; i < host_params.n; i++)
        host_I[i*host_params.lda+i]=1.0;//set diagnal unit 

    checkCudaErrors(cudaMalloc((void **)&(device_I), len));
    checkCudaErrors(cudaMemcpy(device_I, host_I, len, cudaMemcpyHostToDevice));

    // allocate arrays for result checking
    double *device_result;
    double *host_result = (double *)malloc(len);
    checkCudaErrors(cudaMalloc((void **)&(device_result), len));
    checkCudaErrors(cudaMemset(device_result, 0, len));

    cublasHandle_t cb_handle = NULL;
    cudaStream_t stream;
    cublasStatus_t status = cublasCreate_v2(&cb_handle);
    cublasSetPointerMode_v2(cb_handle, CUBLAS_POINTER_MODE_HOST);
    cudaStreamCreate(&stream);
    cublasSetStream_v2(cb_handle, stream);

    double alpha = 1;
    int lda = host_params.lda;//1024
	
	/*modified*/
	double *d_b;
	double *b;
	b = (double *)malloc(sizeof(double)*host_params.piv_len);
	checkCudaErrors(cudaMalloc((void **)&d_b, host_params.piv_len*sizeof(double)));
	cudaMemcpy(d_b, b, host_params.piv_len*sizeof(double), cudaMemcpyHostToDevice);
	b[0]=24;b[1]=30;b[2]=-24;
	//b[0]=2;b[1]=0;b[2]=-1;b[3]=-1;b[4]=2;b[5]=4;

	//fd2=fopen("matrix_b_dense.txt","r");
	//if(!fd2)
	//	printf("fail to open!\n");
	//else
	//{
	//	while(n2<host_params.piv_len)
	//	{
	//		fscanf(fd2,"%lf",&temp2); 
	//		b[n2]=(double)temp2;
	//		n2++;
	//	}
	//}
	//fclose(fd2);

	//for (int i=0; i<100; i++)
	//{
	//	printf("d_LU=%lf\n",b[i]);
	//}
	/*modified*/

	double *d_y;
	double *y;
	y = (double *)malloc(sizeof(double)*piv_len);
	checkCudaErrors(cudaMalloc((void **)&d_y, host_params.piv_len*sizeof(double)));
	cudaMemcpy(d_y, y, host_params.piv_len*sizeof(double), cudaMemcpyHostToDevice);

	double *d_temp;
	double *temp;
	double *x;
	x = (double *)malloc(sizeof(double)*piv_len);
	temp = (double *)malloc(sizeof(double)*piv_len);
	checkCudaErrors(cudaMalloc((void **)&d_temp, host_params.piv_len*sizeof(double)));
	cudaMemcpy(d_temp, temp, host_params.piv_len*sizeof(double), cudaMemcpyHostToDevice);

	checkCudaErrors(cudaMemcpy(host_params.host_LU, host_params.device_LU, len, cudaMemcpyDeviceToHost));

	int i=0;
	y[i]=b[i];
	for(int a1=0;a1<host_params.lda;a1++)
	{
		temp[a1]=0;
		x[a1]=0;
	}
	i++;
	while(i<host_params.lda)
	{
		int n=0;

		while(n<i)
		{
			temp[i]=temp[i]+host_params.host_LU[n*host_params.lda+i]*y[n];
			n++;
		}

		y[i]=b[i]-temp[i];	
		i++;
	}

	//	for(int a1=0;a1<host_params.lda;a1++)
	//{
	//	printf("y=%f\n",y[a1]);
	//}
	for(int a1=0;a1<host_params.lda;a1++)
	{
		temp[a1]=0;
	}

	int j=host_params.lda-1;//2
	x[j]=y[j]/host_params.host_LU[j*host_params.lda+j];
	while(j>0)
	{
		int m=j;//2
		j--;//1
		while(m<host_params.lda)
		{
			temp[j]=temp[j]+host_params.host_LU[m*host_params.lda+j]*x[m];
			m++;
		}
		x[j]=(y[j]-temp[j])/host_params.host_LU[j*host_params.lda+j];
	}

	for(int a2=0;a2<host_params.lda;a2++)
	{
		printf("d_x=%f\n",x[a2]);
	}

	/*modified*/

;
	//LU 是按列排列的

    status = cublasDtrmm(cb_handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, host_params.m, host_params.n, &alpha, host_params.device_LU, lda, device_I, lda, device_result, lda);
	//Result=L*unit. A is a lower symtrical matrix which diagnal is unit
    if (status != CUBLAS_STATUS_SUCCESS)
        errorExit("checkresult: cublas failed");

    status = cublasDtrmm(cb_handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, host_params.m, host_params.n, &alpha, host_params.device_LU, lda, device_result, lda, device_result, lda);
	//Result=Result*U
    if (status != CUBLAS_STATUS_SUCCESS)
        errorExit("checkresult: cublas failed");

    checkCudaErrors(cudaStreamSynchronize(stream));

    // Copy device_LU to host_LU.

    checkCudaErrors(cudaMemcpy(host_params.host_piv, host_params.device_piv, piv_len, cudaMemcpyDeviceToHost));

    // Rebuild the permutation vector.
    int mn = MIN(host_params.m, host_params.n);
    int *perm = (int *) malloc(mn*sizeof(int));

    for (int i = 0 ; i < mn ; ++i)
        perm[i] = i;

    for (int i = 0 ; i < mn ; ++i)
    {
        int j = host_params.host_piv[i];

        if (j >= mn)
            errorExit("Invalid pivot");

        if (i != j)
        {
            int tmp = perm[i];
            perm[i] = perm[j];
            perm[j] = tmp;
        }
    }

    const double tol = 1.0e-6;

    // Verify that L*U = A.
    checkCudaErrors(cudaMemcpy(host_result, device_result, len, cudaMemcpyDeviceToHost));
    bool ok = true;

    for (int i = 0; i < host_params.m; i++)
        for (int j = 0; j < host_params.n; j++)
            if (fabs(host_result[lda*j+i] - host_params.host_A[j*lda + perm[i]]) > tol)
            {
                printf("(%d,%d): found=%f, expected=%f\n", i, j, host_result[lda*j+i], host_params.host_A[j*lda + perm[i]]);
                ok = false;
                break;
            }


	
	free(b);
	free(y);
	free(x);
	free(temp);
    free(perm);
    free(host_I);
    free(host_result);
	checkCudaErrors(cudaFree(d_y));
	checkCudaErrors(cudaFree(d_temp));
	checkCudaErrors(cudaFree(d_b));
    checkCudaErrors(cudaFree(device_I));
    checkCudaErrors(cudaFree(device_result));
    printf("done\n");
    return ok;
}

bool launch_test(Parameters &host_params)
{
    memsetup(host_params);
    launch(host_params);
    bool result = checkresult(host_params);
    finalize(host_params);
    return result;
}

void print_usage(const char *exec_name)
{
    printf("Usage: %s -matrix_size=N <-device=N>(optional)\n", exec_name);
    printf("  matrix_size: the size of a NxN matrix. It must be greater than 0.\n");
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
#if CUDART_VERSION < 5000
#error cdpLU requires CUDA 5.0 to run, waiving testing...
#endif

    printf("Starting LU Decomposition (CUDA Dynamic Parallelism)\n");

    //int matrix_size = 1024;
	int matrix_size = 3;

    if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
        checkCmdLineFlag(argc, (const char **)argv, "h"))
    {
        print_usage(argv[0]);
        exit(EXIT_SUCCESS);
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "matrix_size"))
    {
        matrix_size = getCmdLineArgumentInt(argc, (const char **)argv, "matrix_size");

        if (matrix_size <= 0)
        {
            printf("Invalid matrix size given on the command-line: %d\n", matrix_size);
            exit(EXIT_FAILURE);
        }
    }
    else if (argc > 3)
    {
        print_usage(argv[0]);
        exit(EXIT_SUCCESS);
    }

    // The test requires CUDA 5 or greater.
    // The test requires an architecture SM35 or greater (CDP capable).
    int cuda_device = findCudaDevice(argc, (const char **)argv);
    cudaDeviceProp deviceProps;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProps, cuda_device));
    int cdpCapable = (deviceProps.major == 3 && deviceProps.minor >= 5) || deviceProps.major >=4;

    printf("GPU device %s has compute capabilities (SM %d.%d)\n", deviceProps.name, deviceProps.major, deviceProps.minor);

    if (!cdpCapable)
    {
        printf("cdpLUDecomposition requires SM 3.5 or higher to use CUDA Dynamic Parallelism.  Exiting...\n");
        cudaDeviceReset();
        exit(EXIT_SUCCESS);
    }

    Parameters host_params;
    memset(&host_params, 0, sizeof(Parameters));
    set_defaults(host_params, matrix_size);

    printf("Compute LU decomposition of a random %dx%d matrix using CUDA Dynamic Parallelism\n", matrix_size, matrix_size);
    printf("Launching single task from device...\n");
    bool result = launch_test(host_params);

    cudaDeviceReset();

    if (result)
    {
        printf("Tests suceeded\n");
        exit(EXIT_SUCCESS);
    }
    else
    {
        exit(EXIT_FAILURE);
    }
}
