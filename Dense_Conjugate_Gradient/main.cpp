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
/*
void genTridiag(int *I, int *J, float *val, int N, int nz)
{
    I[0] = 0, J[0] = 0, J[1] = 1;//第一行信息在此定义
    val[0] = ((float)rand()/RAND_MAX)+10;	//产生10到11的随机数
    val[1] = ((float)rand()/RAND_MAX);			//产生0到10的随机数
	 //val[0] = -N;
	 //val[1] = 1;
    int start;

    for (int i = 1; i < N; i++)				//对每一行进行循环
    {
        if (i > 1)
        {
            I[i] = I[i-1]+3;
        }
        else
        {
            I[1] = 2;
        }
		//对每一行非零个数进行设置

        start = (i-1)*3 + 2;//没有对第一行进行讨论
        J[start] = i - 1;
        J[start+1] = i;

        if (i < N-1)
        {
            J[start+2] = i + 1;
        }
		//如果不是最后一行，每一行都有三个非零数，所以都是列号

        val[start] = val[start-1];
        val[start+1] = ((float)rand()/RAND_MAX)+10;
		//val[start+1] = -N;

        if (i < N-1)
        {
            val[start+2] = ((float)rand()/RAND_MAX);
			// val[start+2] = 1;
        }
    }

    I[N] = nz;//nz代表这个矩阵中所以非零数的个数
}//最终结果是对角线上产生的都是10到11的随机数，对角线旁边都是0到1的随机数

*/


int main(int argc, char **argv)
{
	
    int M = 0, N = 0, nz = 0, *I = NULL, *J = NULL;
    float *val = NULL;
    const float tol = 0.00005;//精度
    const int max_iter = 100000000;//最高迭代次数
    float *x,*rhs;
    float a, b, na, r0,r1;
    int *d_col, *d_row;
    float *d_val, *d_x, dot;
    float *d_p, *d_Ax,*d_r;
    int k;
    float alpha, beta, alpham1;
	int n1=0,n2=0,n3=0,n4=0;
	int temp1,temp2;
	float temp3;
	long double temp4;
	float *A=NULL;
	float *d_A;


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

    /* Generate a random tridiagonal symmetric matrix in CSR format */
    M = N = 100;						//N是行数
   // nz = (N-2)*3 + 4;						//nz=13
	nz=N*N;
    I = (int *)malloc(sizeof(int)*(N+1));	//I就代表csr模型中的csrRowPtrA，需要分配空间N+1即可
    J = (int *)malloc(sizeof(int)*nz);		//J就代表csr模型中的csrColIndA，需要分配空间NZ
    val = (float *)malloc(sizeof(float)*nz);	//val代表csr模型中所以非零数据的值
	A = (float *)malloc(sizeof(float)*nz);
	//I[0]=0;I[1]=2;I[2]=5;I[3]=7;
	//J[0]=0;J[1]=1;J[2]=0;J[3]=1;J[4]=2;J[5]=1;J[6]=2;
	//val[0]=4;val[1]=3;val[2]=3;val[3]=4;val[4]=-1;val[5]=-1;val[6]=4;
	

    //genTridiag(I, J, val, N, nz);  //随机产生行数为N的三角对称矩阵

	//printf("val1=%f\n",val[0]);
	//printf("val2=%f\n",val[1]);
	//printf("val3=%f\n",val[2]);
	//printf("val4=%f\n",val[3]);
	//printf("val5=%f\n",val[4]);

	//open file
	fd3=fopen("Matrix_val_dense.txt","r");
	if(!fd3)
	printf("fail to open!\n");
	else
	{
	while(n3<nz)
	{
		fscanf(fd3,"%f",&temp3); 
		//printf("%f\n",temp3);
		A[n3]=temp3;
		n3++;
   }
	}
	 //for (int i=0;i<nz;i++)
    // printf("%f\n",A[i]);
	fclose(fd3);


    x = (float *)malloc(sizeof(float)*N);
    rhs = (float *)malloc(sizeof(float)*N);
	//rhs[0]=24;rhs[1]=30;rhs[2]=-24;
	
    for (int i = 0; i < N; i++)
    {
        //rhs[i] = 1.0;
        x[i] = 0.0;
    }
	
	//open file
	fd4=fopen("Matrix_b_dense.txt","r");
	if(!fd4)
	printf("fail to open!\n");
	else
	{
	while(n4<N)
	{
		fscanf(fd4,"%Lf",&temp4); 
		//printf("%Lf\n",temp4);
		rhs[n4]=(float)temp4;
		n4++;
   }
	}
	/*for (int i=0;i<N;i++)
		printf("%f\n",rhs[i]);*/
	fclose(fd4);

	time_t begin,end; 
	begin=clock(); 
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

    checkCudaErrors(cudaMalloc((void **)&d_col, nz*sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_row, (N+1)*sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_val, nz*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_x, N*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_r, N*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_p, N*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_Ax, N*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_A, nz*sizeof(float)));//for dense matrix

    cudaMemcpy(d_col, J, nz*sizeof(int), cudaMemcpyHostToDevice);//将每一个数值从host拷到device。列号从host拷到device
    cudaMemcpy(d_row, I, (N+1)*sizeof(int), cudaMemcpyHostToDevice);//行号从host拷到device
    cudaMemcpy(d_val, val, nz*sizeof(float), cudaMemcpyHostToDevice);//矩阵值从host拷到device
    cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);//值都为0
    cudaMemcpy(d_r, rhs, N*sizeof(float), cudaMemcpyHostToDevice);//值都为1。其实就是b
	cudaMemcpy(d_A, A, N*N*sizeof(float), cudaMemcpyHostToDevice);//copy the value from host to device. densematrix


    alpha = 1.0;
    alpham1 = -1.0;
    beta = 0.0;
    r0 = 0.;
	//float *temp;
	//temp=(float*)malloc(N*sizeof(float));
	float *out1;
	out1=(float*)malloc(N*sizeof(float));

	cublasSgemv (cublasHandle, CUBLAS_OP_N, N,N,&alpha, d_A, N, d_x, 1, &beta,d_Ax, 1);  
    //cusparseScsrmv(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &alpha, descr, d_val, d_row, d_col, d_x, &beta, d_Ax);//产生的矩阵行数和列数一定是相同的。d_x就是CG算法中的x，d_Ax就是CG算法中的y。会计算y的值，但是返回值是状态
	//上述函数输出结果为0。实际计算为d_Ax=A*d_x
	//printf("d_Ax=%f\n",d_Ax);//输出结果为0
	/*cudaMemcpy(out1, d_Ax, N*sizeof(float), cudaMemcpyDeviceToHost);
	for(int a1=0;a1<N;a1++)
	{
			printf("d_Ax=%f\n",out1[a1]);
	}*/
    
	cublasSaxpy(cublasHandle, N, &alpham1, d_Ax, 1, d_r, 1);
	//完成的计算是d_r=d_r+alpham1*d_Ax
	//cudaMemcpy(temp, d_r, N*sizeof(float), cudaMemcpyDeviceToHost);
	//printf("d_r=%f\n",temp[2]);//输出结果为长度为N的数组，全部是1
	//r=b-A*x;其实就是b
    
	cublasStatus = cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);
	//printf("r1=%f\n",r1);//输出结果为N
	/*上面函数主要实现dr与dr的点乘。就是matlab函数中的q0*/
    k = 1;

    while (r1 > tol && k <= max_iter)
    {
        if (k > 1)
        {
            b = r1 / r0;//s=q/q0;
            cublasStatus = cublasSscal(cublasHandle, N, &b, d_p, 1);//返回值为dp=b*dp。s*p
   			cublasStatus = cublasSaxpy(cublasHandle, N, &alpha, d_r, 1, d_p, 1);//返回值为dp=dr+dp
        }
        else
        {
            cublasStatus = cublasScopy(cublasHandle, N, d_r, 1, d_p, 1);//把dr的值拷贝到dp中。dp就变成了全部为1的vector
        }

		cublasSgemv (cublasHandle, CUBLAS_OP_N, N,N,&alpha, d_A, N, d_p, 1, &beta,d_Ax, 1); 
        //cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &alpha, descr, d_val, d_row, d_col, d_p, &beta, d_Ax);//与上一次调用唯一不同的地方时，这里的x变成了d_p不再是d_x
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
        printf("iteration = %3d\n",k );
		//printf("residual =%e\n",sqrt(r1));
        k++;
    }
	
	cudaMemcpy(out1, d_x, N*sizeof(float), cudaMemcpyDeviceToHost);
	for(int a1=0;a1<N;a1++)
	{
			printf("d_Ax=%f\n",out1[a1]);
	}
	
    cudaMemcpy(x, d_x, N*sizeof(float), cudaMemcpyDeviceToHost);

   /* float rsum, diff, err = 0.0;

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
    }*/

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
	end=clock(); 
	printf("runtime:%f \n",double(end-begin)/CLOCKS_PER_SEC); 
   // printf("Test Summary:  Error amount = %f\n", err);
    exit((k <= max_iter) ? 0 : 1);
	
}
