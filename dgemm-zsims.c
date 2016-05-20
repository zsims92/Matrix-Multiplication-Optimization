/*
Please include compiler name below (you may also include any other modules you would like to be loaded)

COMPILER= gnu

Please include All compiler flags and libraries as you want them run. You can simply copy this over from the Makefile's first few lines

CC = cc
OPT = -O3 -march=barcelona -msse2 -msse3 -m3dnow -mfpmath=sse -fomit-frame-pointer -funroll-loops -ffast-math -ftree-vectorize
CFLAGS = -Wall -std=gnu99 $(OPT)
MKLROOT = /opt/intel/composer_xe_2013.1.117/mkl
LDLIBS = -lrt -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm

*/
#include <emmintrin.h>

const char* dgemm_desc = "Blocking with SSE by Zachary Sims";
#define RS_M 2	//M stride length
#define RS_K 2	//K stride length
#define RS_N 6	//N stride length
#define IsT 2	///I step value
#define even(M) (((M)%2)?((M)+1):(M))	//Make even
#define min(a,b) (((a)<(b))?(a):(b))		

static void do_block(int M, int K, int N, int MM, int KK, double* A, double* B, double* C) {

    int jstep;
    __m128d a0,a1,b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,c0,c1,d0,d1;

    for (int k = 0; k < K; k += RS_K ) {
        for (int j = 0; j < N; j += RS_N) {
            jstep = min (RS_N,N-j);
			switch(jstep){	//Handles cases where N is not divisible by 6. 2 and 4 because the value will always be even
				case 2:
					b0 = _mm_load1_pd(B+k+j*KK);
					b1 = _mm_load1_pd(B+k+1+j*KK);
					b2 = _mm_load1_pd(B+k+(j+1)*KK);
					b3 = _mm_load1_pd(B+k+1+(j+1)*KK);
					for (int i = 0; i < M; i += RS_M) {
						a0 = _mm_load_pd(A+i+k*MM);
						a1 = _mm_load_pd(A+i+(k+1)*MM);

						c0 = _mm_load_pd(C+i+j*MM);
						c1 = _mm_load_pd(C+i+(j+1)*MM);

						d0 = _mm_add_pd(c0, _mm_mul_pd(a0,b0));
						d1 = _mm_add_pd(c1, _mm_mul_pd(a0,b2));
						c0 = _mm_add_pd(d0, _mm_mul_pd(a1,b1));
						c1 = _mm_add_pd(d1, _mm_mul_pd(a1,b3));
						_mm_store_pd(C+i+j*MM,c0);
						_mm_store_pd(C+i+(j+1)*MM,c1);
					}
					break;
				case 4:
					b0 = _mm_load1_pd(B+k+j*KK);
					b1 = _mm_load1_pd(B+k+1+j*KK);
					b2 = _mm_load1_pd(B+k+(j+1)*KK);
					b3 = _mm_load1_pd(B+k+1+(j+1)*KK);
					b4 = _mm_load1_pd(B+k+(j+2)*KK);
					b5 = _mm_load1_pd(B+k+1+(j+2)*KK);
					b6 = _mm_load1_pd(B+k+(j+3)*KK);
					b7 = _mm_load1_pd(B+k+1+(j+3)*KK);
					for (int i = 0; i < M; i += RS_M) {
						a0 = _mm_load_pd(A+i+k*MM);
						a1 = _mm_load_pd(A+i+(k+1)*MM);

						c0 = _mm_load_pd(C+i+j*MM);
						d0 = _mm_add_pd(c0, _mm_mul_pd(a0,b0));
						c0 = _mm_add_pd(d0, _mm_mul_pd(a1,b1));
						_mm_store_pd(C+i+j*MM,c0);

						c0 = _mm_load_pd(C+i+(j+1)*MM);
						d0 = _mm_add_pd(c0, _mm_mul_pd(a0,b2));
						c0 = _mm_add_pd(d0, _mm_mul_pd(a1,b3));
						_mm_store_pd(C+i+(j+1)*MM,c0);

						c0 = _mm_load_pd(C+i+(j+2)*MM);
						d0 = _mm_add_pd(c0, _mm_mul_pd(a0,b4));
						c0 = _mm_add_pd(d0, _mm_mul_pd(a1,b5));
						_mm_store_pd(C+i+(j+2)*MM,c0);

						c0 = _mm_load_pd(C+i+(j+3)*MM);
						d0 = _mm_add_pd(c0, _mm_mul_pd(a0,b6));
						c0 = _mm_add_pd(d0, _mm_mul_pd(a1,b7));
						_mm_store_pd(C+i+(j+3)*MM,c0);
					}
					break;
				case 6:
					b0 = _mm_load1_pd(B+k+j*KK);
					b1 = _mm_load1_pd(B+k+1+j*KK);
					b2 = _mm_load1_pd(B+k+(j+1)*KK);
					b3 = _mm_load1_pd(B+k+1+(j+1)*KK);
					b4 = _mm_load1_pd(B+k+(j+2)*KK);
					b5 = _mm_load1_pd(B+k+1+(j+2)*KK);
					b6 = _mm_load1_pd(B+k+(j+3)*KK);
					b7 = _mm_load1_pd(B+k+1+(j+3)*KK);
					b8 = _mm_load1_pd(B+k+(j+4)*KK);
					b9 = _mm_load1_pd(B+k+1+(j+4)*KK);
					b10 = _mm_load1_pd(B+k+(j+5)*KK);
					b11 = _mm_load1_pd(B+k+1+(j+5)*KK);
					for (int i = 0; i < M; i += RS_M) {
						a0 = _mm_load_pd(A+i+k*MM);
						a1 = _mm_load_pd(A+i+(k+1)*MM);

						c0 = _mm_load_pd(C+i+j*MM);
						d0 = _mm_add_pd(c0, _mm_mul_pd(a0,b0));
						c0 = _mm_add_pd(d0, _mm_mul_pd(a1,b1));
						_mm_store_pd(C+i+j*MM,c0);

						c0 = _mm_load_pd(C+i+(j+1)*MM);
						d0 = _mm_add_pd(c0, _mm_mul_pd(a0,b2));
						c0 = _mm_add_pd(d0, _mm_mul_pd(a1,b3));
						_mm_store_pd(C+i+(j+1)*MM,c0);

						c0 = _mm_load_pd(C+i+(j+2)*MM);
						d0 = _mm_add_pd(c0, _mm_mul_pd(a0,b4));
						c0 = _mm_add_pd(d0, _mm_mul_pd(a1,b5));
						_mm_store_pd(C+i+(j+2)*MM,c0);

						c0 = _mm_load_pd(C+i+(j+3)*MM);
						d0 = _mm_add_pd(c0, _mm_mul_pd(a0,b6));
						c0 = _mm_add_pd(d0, _mm_mul_pd(a1,b7));
						_mm_store_pd(C+i+(j+3)*MM,c0);

						c0 = _mm_load_pd(C+i+(j+4)*MM);
						d0 = _mm_add_pd(c0, _mm_mul_pd(a0,b8));
						c0 = _mm_add_pd(d0, _mm_mul_pd(a1,b9));
						_mm_store_pd(C+i+(j+4)*MM,c0);

						c0 = _mm_load_pd(C+i+(j+5)*MM);
						d0 = _mm_add_pd(c0, _mm_mul_pd(a0,b10));
						c0 = _mm_add_pd(d0, _mm_mul_pd(a1,b11));
						_mm_store_pd(C+i+(j+5)*MM,c0);
					}
					break;
			}
        }
    }
}

//Copies array values to a new temp array.
static double* copyOut(int lda, int M, int N, double* A, double* newA) {
    int newM = even(M);
    int newN = even(N);
    int istep;
    __m128d a;

    for (int j = 0; j < N; j++) {
        for (int i = 0; i < M; i+= IsT) {
            istep = min(IsT, M-i);
			if (istep==1) {
                newA[i+j*newM] = A[i+j*lda];
            } else {
                a = _mm_loadu_pd(A+i+j*lda);
                _mm_store_pd(newA+i+j*newM,a);
            }
        }
    }

    if (N%2) {
        for (int i = 0; i < newM; i++) {
            newA[i+(newN-1)*newM] = 0.0;
        }
    }

    return newA;
}

//Place calculated values back in to an array
static void addIn(double* newA, double*  A, int M, int N, int lda, int newM) {
    __m128d a;
    int istep;
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < M; i+= IsT) {
            istep = min(IsT,M-i);
            if (istep == 1) {
                A[i+j*lda] = newA[i+j*newM];
            } else {
                a = _mm_load_pd(newA + i + j*newM);
                _mm_storeu_pd(A+i+j*lda,a);
            }
        }
    }
}

void square_dgemm (int lda, double* A, double* B, double* C)
{
    int bs_row = 222;		//Found from testing
    int bs_col = 12;		//Found from testing
    int bs_inner =  222;	//Found from testing
    int newM, newK, newN;

    double newA[50000];		//Temp array sizes that hold all values.
    double newB[200000];
    double newC[8000];

    for (int k = 0; k < lda; k += bs_inner) {
        int K = min (bs_inner, lda-k);
        copyOut(lda, K, lda, B+k, newB);
        newK = even(K);
        for (int i = 0; i < lda; i += bs_row) {
            int M = min (bs_row, lda-i);

            copyOut(lda, M, K, A+i+k*lda, newA);
            newM = even(M);

            for (int j = 0; j < lda; j += bs_col) {
                int N = min (bs_col, lda-j);
                newN = even(N);
                copyOut(lda, M, N, C+i+j*lda, newC);
                do_block(newM,newK,newN,newM,newK, newA, newB+j*newK, newC);
                addIn(newC, C+i+j*lda, M, N, lda, newM);
            }
        }
    }
}