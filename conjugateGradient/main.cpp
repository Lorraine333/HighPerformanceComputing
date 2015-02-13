#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include<ctime> 
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>


#include <helper_functions.h>  // helper for shared functions common to CUDA SDK samples
#include <helper_cuda.h>       // helper function CUDA error checking and intialization
using namespace std;


const char *sSDKname     = "conjugateGradient";
FILE   *fd1; 
FILE   *fd2;
FILE   *fd3;
FILE   *fd4;


int main(int argc, char **argv)
{

	int M = 0, N = 0, nz = 0, *I = NULL, *J = NULL;
	float *val = NULL;
	//const float tol = 0.0000005;//精度
	const float tol = 1e-12f;
	const int max_iter = 100000000;//最高迭代次数
	float *x;
	float *rhs;
	float a, b, na, r0, r1;
	int *d_col, *d_row;
	float *d_val, *d_x, dot;
	float *d_r, *d_p, *d_Ax;
	int k;
	float alpha, beta, alpham1;
	int n1=0,n2=0,n3=0,n4=0;
	int temp1,temp2;
	float temp3,temp4;

	// This will pick the best possible CUDA capable device
	cudaDeviceProp deviceProp;
	int devID = findCudaDevice(argc, (const char **)argv);

	if (devID < 0)
	{
		printf("exiting...\n");//if there is a device of GPU, return a negative number
		exit(EXIT_SUCCESS);
	}

	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));


	printf("> GPU device has %d Multi-Processors, SM %d.%d compute capabilities\n\n",
		deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

	int version = (deviceProp.major * 0x10 + deviceProp.minor);

	if (version < 0x11)
	{
		printf("%s: requires a minimum CUDA compute 1.1 capability\n", sSDKname);
		cudaDeviceReset();
		exit(EXIT_SUCCESS);
	}

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop); 
	/* Generate a random tridiagonal symmetric matrix in CSR format */
	M = N =5001;						//N是行数
	// nz = (N-2)*3 + 4;						//nz=13
	nz=15195;
	I = (int *)malloc(sizeof(int)*(N+1));	//I就代表csr模型中的csrRowPtrA，需要分配空间N+1即可
	J = (int *)malloc(sizeof(int)*nz);		//J就代表csr模型中的csrColIndA，需要分配空间NZ
	val = (float *)malloc(sizeof(float)*nz);	//val代表csr模型中所以非零数据的值
	//I[0]=0;I[1]=2;I[2]=5;I[3]=7;
	//J[0]=0;J[1]=1;J[2]=0;J[3]=1;J[4]=2;J[5]=1;J[6]=2;
	//val[0]=4;val[1]=3;val[2]=3;val[3]=4;val[4]=-1;val[5]=-1;val[6]=4;


	//open the file

	fd1=fopen("Matrix_I_sparse.txt","r");
	if(!fd1)
		printf("fail to open!\n");
	else
	{
		while(n1<N+1)
		{
			fscanf(fd1,"%d",&temp1); 
			//printf("%d\n",temp1);
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
			//printf("%d\n",temp2);
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



	x = (float *)malloc(sizeof(float)*N);
	rhs = (float *)malloc(sizeof(float)*N);
	//rhs[0]=24;rhs[1]=30;rhs[2]=-24;



	
	/* Get handle to the CUBLAS context */
	cublasHandle_t cublasHandle = 0;
	cublasStatus_t cublasStatus;
	cublasStatus = cublasCreate(&cublasHandle);//初始化cublasHandle
	checkCudaErrors(cublasStatus);

	/* Get handle to the CUSPARSE context */
	cusparseHandle_t cusparseHandle = 0;				//一个指针，指向一个不透明的cusparse变量
	cusparseStatus_t cusparseStatus;					//一个枚举型变量，包括好多状态
	cusparseStatus = cusparseCreate(&cusparseHandle);	//初始化cusparseHandle
	checkCudaErrors(cusparseStatus);					//检测错误。这个函数在helper_cuda.h中定义

	cusparseMatDescr_t descr = 0;				//一个用来表示矩阵shape和properties的struct（matrix discription）
	cusparseStatus = cusparseCreateMatDescr(&descr);//初始化matrix discription
	checkCudaErrors(cusparseStatus);

	cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);//设置matrix type的field
	cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);//设置matrix indexbase。从0开始计数

	cudaEventRecord(start,0);

	checkCudaErrors(cudaMalloc((void **)&d_col, nz*sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&d_row, (N+1)*sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&d_val, nz*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_x, N*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_r, N*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_p, N*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_Ax, N*sizeof(float)));

	int times=0;
	while(times<500)
	{

		for (int i = 0; i < N; i++)
		{
			rhs[i] = 1.0;
			x[i] = 0.0;
		}

	cudaMemcpy(d_col, J, nz*sizeof(int), cudaMemcpyHostToDevice);//将每一个数值从host拷到device。列号从host拷到device
	cudaMemcpy(d_row, I, (N+1)*sizeof(int), cudaMemcpyHostToDevice);//行号从host拷到device
	cudaMemcpy(d_val, val, nz*sizeof(float), cudaMemcpyHostToDevice);//矩阵值从host拷到device
	cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);//值都为0
	cudaMemcpy(d_r, rhs, N*sizeof(float), cudaMemcpyHostToDevice);//值都为1。其实就是b


	alpha = 1.0;
	alpham1 = -1.0;
	beta = 0.0;
	r0 = 0.;
	float *temp;
	temp=(float*)malloc(N*sizeof(float));
	float *out1;
	out1=(float*)malloc(N*sizeof(float));



	cusparseScsrmv(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &alpha, descr, d_val, d_row, d_col, d_x, &beta, d_Ax);//产生的矩阵行数和列数一定是相同的。d_x就是CG算法中的x，d_Ax就是CG算法中的y。会计算y的值，但是返回值是状态
	//上述函数输出结果为0。实际计算为d_Ax=A*d_x
	//printf("d_Ax=%f\n",d_Ax);//输出结果为0

	cublasSaxpy(cublasHandle, N, &alpham1, d_Ax, 1, d_r, 1);
	//完成的计算是d_r=d_r+alpham1*d_Ax
	//cudaMemcpy(temp, d_r, N*sizeof(float), cudaMemcpyDeviceToHost);
	//printf("d_r=%f\n",temp[2]);//输出结果为长度为N的数组，全部是1
	//r=b-A*x;其实就是b

	cublasStatus = cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);
	//printf("r1=%f\n",r1);//输出结果为N
	/*上面函数主要实现dr与dr的点乘。就是matlab函数中的q0*/
	k = 0;

	while (r1 > tol*tol && k <= max_iter)
		//while (k==1)
	{
		k++;
		if (k > 1)
		{
			b = r1 / r0;//s=q/q0;
			//printf("d_p=%f\n",b);
			cublasStatus = cublasSscal(cublasHandle, N, &b, d_p, 1);//返回值为dp=b*dp。s*p
			cublasStatus = cublasSaxpy(cublasHandle, N, &alpha, d_r, 1, d_p, 1);//返回值为dp=dr+dp

		}

		else
		{
			cublasStatus = cublasScopy(cublasHandle, N, d_r, 1, d_p, 1);//把dr的值拷贝到dp中。dp就变成了全部为1的vector

		}

		cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &alpha, descr, d_val, d_row, d_col, d_p, &beta, d_Ax);//与上一次调用唯一不同的地方时，这里的x变成了d_p不再是d_x
		//完成计算d_Ax=d_p*A。结果就是matlab中的w

		cublasStatus = cublasSdot(cublasHandle, N, d_p, 1, d_Ax, 1, &dot);//点乘。dp和d_Ax的点乘，dot是取值result。q1=p'*w;
		a = r1 / dot;//t=q0/q1;

		cublasStatus = cublasSaxpy(cublasHandle, N, &a, d_p, 1, d_x, 1);//d_x=a*dp+dx
		//x=x+t*p;最后结果！！！！




		na = -a;
		cublasStatus = cublasSaxpy(cublasHandle, N, &na, d_Ax, 1, d_r, 1);//d_r=na*dAx+dr
		//r=r-t*w;

		r0 = r1;
		cublasStatus = cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);//dr点乘dr，结果放在r1中
		cudaThreadSynchronize();

	}

	//cudaMemcpy(out1, d_x, N*sizeof(float), cudaMemcpyDeviceToHost);
	//for(int a1=0;a1<N;a1++)
	//{
	//printf("d_Ax=%f\n",out1[a1]);
	//}

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);

	cudaMemcpy(x, d_x, N*sizeof(float), cudaMemcpyDeviceToHost);
	printf("iteration = %3d\n",k );
	printf("residual =%e\n",sqrt(r1));

	float time;
	cudaEventElapsedTime(&time, start, stop);
	std::cout << "Elapsed Time: " << time << " ms" << std::endl;

	times=times+1;
	}

	float rsum, diff, err = 0.0;

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

	cusparseDestroy(cusparseHandle);
	cublasDestroy(cublasHandle);

	free(I);
	free(J);
	free(val);
	free(x);
	free(rhs);
	cudaFree(d_col);
	cudaFree(d_row);
	cudaFree(d_val);
	cudaFree(d_x);
	cudaFree(d_r);
	cudaFree(d_p);
	cudaFree(d_Ax);

	cudaDeviceReset();
	// printf("Test Summary:  Error amount = %f\n", err);
	exit((k <= max_iter) ? 0 : 1);

}
