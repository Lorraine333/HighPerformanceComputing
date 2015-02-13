#ifndef _CDP_LU_H_
#define _CDP_LU_H_

extern "C" {

    // Parameters which are passed to a test algorithm
    typedef struct Parameters_s
    {
        int m, n, lda;
        double *host_A;    /* Original matrix. */
        double *host_LU;   /* LU decomposition. */
        int *host_piv;     /* The pivots. */
        double *device_A;  /* Original matrix. */
        double *device_LU; /* LU decomposition. */
        int *device_piv;   /* The pivots. */
        size_t data_size;  /* Size of each data element. */
        int data_len;      /* Number of elements in matrix. */
        int piv_len;       /* Number of elements in returned pivot array. */
        unsigned long long flop_count;
        int seed;          /* Seed to initialize the random number generator. */

        int  host_info;    /* The result of the algorithm. */
        int *device_info;  /* The result of the algorithm. */

    } Parameters;

    void dgetrf_test(Parameters *host_params, Parameters *device_params);

} /* extern "C" */

#endif /* !defined _CDP_LU_H_ */
