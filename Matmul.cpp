/*
Author: Aniruddha Bala
Roll number: 15655
MTech CDS
*/
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<string.h>
#include<algorithm>
#include<assert.h>
using namespace std;

#define MAXN 2049
#define MAXV 1000
#define OPT_BLCK_SZ 512

double A[MAXN][MAXN],B[MAXN][MAXN], C[MAXN][MAXN];

struct timespec start, finish;

//without any cache optimization
void matmul_v1(double a[][MAXN], double b[][MAXN], double c[][MAXN], int n){
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            for(int k=0; k<n; k++){
                c[i][j] += a[i][k]*b[k][j];
            }
        }
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &finish);
    double run_time = (finish.tv_sec - start.tv_sec) + (finish.tv_nsec - start.tv_nsec)/1e9;
    printf("Naive Matrix multiplication without any cache optimization : %lf secs\n",run_time);
}


// exploiting temporal locality
void matmul_v2(double a[][MAXN], double b[][MAXN], double c[][MAXN], int n){
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            double sum = 0;
            for(int k=0; k<n; k++){
                sum += a[i][k]*b[k][j];
            }
            c[i][j] = sum;
        }
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &finish);
    double run_time = (finish.tv_sec - start.tv_sec) + (finish.tv_nsec - start.tv_nsec)/1e9;
    printf("With temporal locality: %lf secs\n",run_time);
}

// with loop interchange
void matmul_v3(double a[][MAXN], double b[][MAXN], double c[][MAXN], int n){
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
    for(int i=0; i<n; i++){
        for(int k=0; k<n; k++){
            double x = a[i][k];
            for(int j=0; j<n; j++){
                c[i][j] += x*b[k][j];
            }
        }
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &finish);
    double run_time = (finish.tv_sec - start.tv_sec) + (finish.tv_nsec - start.tv_nsec)/1e9;
    printf("With loop interchange: %lf secs\n",run_time);
}

//with loop unrolling
void matmul_v4(double a[][MAXN], double b[][MAXN], double c[][MAXN], int n){
    
    assert(n%2==0);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
    for(int i=0; i<n; i+=2){
        for(int k=0; k<n; k+=2){
            double x = a[i][k];
            double y = a[i][k+1];
            double u = a[i+1][k];
            double v = a[i+1][k+1];
            for(int j=0; j<n; j+=2){
                c[i][j] += x*b[k][j];
                c[i][j+1] += x*b[k][j+1];
                c[i][j] += y*b[k+1][j];
                c[i][j+1] += y*b[k+1][j+1];

                c[i+1][j] += u*b[k][j];
                c[i+1][j+1] += u*b[k][j+1];
                c[i+1][j] += v*b[k+1][j];
                c[i+1][j+1] += v*b[k+1][j+1];

            }

        }
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &finish);
    double run_time = (finish.tv_sec - start.tv_sec) + (finish.tv_nsec - start.tv_nsec)/1e9;
    printf("With loop unrolling: %lf secs\n",run_time);
}


//using block multiplication
void matmul_v5(double a[][MAXN], double b[][MAXN], double c[][MAXN], int n, int blck_sz){
 
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
    for(int bk=0; bk<n; bk+=blck_sz){
        for(int bj=0; bj<n; bj+=blck_sz){
            int to_k = min(bk+blck_sz,n);
            int to_j = min(bj+blck_sz,n);
            for(int i=0; i<n; i++){
                for(int k=bk; k<to_k; k++){
                    double x = a[i][k];
                    for(int j=bj; j<to_j; j++)
                        c[i][j]+=x*b[k][j]; 
                }
            }
        }
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &finish);
    double run_time = (finish.tv_sec - start.tv_sec) + (finish.tv_nsec - start.tv_nsec)/1e9;
    printf("With blocking block size %d: %lf secs\n",blck_sz,run_time);
}

//using block multiplication with loop unrolling
void matmul_v6(double a[][MAXN], double b[][MAXN], double c[][MAXN], int n, int blck_sz){

    assert(n%2==0);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
    for(int bk=0; bk<n; bk+=blck_sz){
        for(int bj=0; bj<n; bj+=blck_sz){
            int to_k = min(bk+blck_sz,n);
            int to_j = min(bj+blck_sz,n);
            for(int i=0; i<n; i+=2){
                for(int k=bk; k<to_k; k+=2){
                    double x = a[i][k];
                    double y = a[i][k+1];
                    double u = a[i+1][k];
                    double v = a[i+1][k+1];
                    for(int j=bj; j<to_j; j+=2)
                    {
                        c[i][j] += x*b[k][j];
                        c[i][j+1] += x*b[k][j+1];
                        c[i][j] += y*b[k+1][j];
                        c[i][j+1] += y*b[k+1][j+1];

                        c[i+1][j] += u*b[k][j];
                        c[i+1][j+1] += u*b[k][j+1];
                        c[i+1][j] += v*b[k+1][j];
                        c[i+1][j+1] += v*b[k+1][j+1];
                    }
 
                }
            }
        }
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &finish);
    double run_time = (finish.tv_sec - start.tv_sec) + (finish.tv_nsec - start.tv_nsec)/1e9;
    printf("With blocking and loop unrolling (block size %d) : %lf secs\n",blck_sz,run_time);
}
void read_mat(double a[][MAXN], int n){
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            int x = scanf("%lf", &a[i][j]);
        }
    }
}

void rand_mat(double a[][MAXN], int n){
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            a[i][j] = (rand()%10000)/100.0;
        }
    }
}
void print_mat(double a[][MAXN], int n){  
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            printf("%lf ", a[i][j]);
        }
        printf("\n");
    }
}

int main(){
    int n;
    printf("Enter dimension for square matrix (n):");
    int x = scanf("%d",&n);
    rand_mat(A,n);
    rand_mat(B,n);
    memset(C, 0.0, sizeof(C));
    matmul_v1(A,B,C,n);
    memset(C, 0.0, sizeof(C));
    matmul_v2(A,B,C,n);
    memset(C, 0.0, sizeof(C));    
    matmul_v3(A,B,C,n);
    memset(C, 0.0, sizeof(C));    
    matmul_v4(A,B,C,n);
    //for(int b = 16; b<=1024; b=b<<1){
    memset(C, 0.0, sizeof(C));   
    matmul_v5(A,B,C,n,OPT_BLCK_SZ);
    //}
    memset(C, 0.0, sizeof(C));   
    matmul_v6(A,B,C,n,OPT_BLCK_SZ);
   // print_mat(C,n);
}
