#include "matrixOperations.h"

// VECTOR FUNCTIONS
#ifdef USE_SUITESPARSE

double * onesv(int n)
{
    /* x = ones(n, 1) */
    return repmatv(1.0, n);
}

double * zerosv(int n)
{
    /* x = zeros(n, 1) */
    return repmatv(0.0, n);
}

double * repmatv(double a, int n)
{
    /* x = repmat(a, n, 1) */
    int i;
    double *y;

    y = (double *)malloc(n * sizeof(*y));
    if (y == NULL) return y;

    for (i = 0; i < n; i++)
    {
        y[i] = a;
    }

    return y;
}

double * copyv(double *x, int n)
{
    double *y;
    y = (double *)malloc(n * sizeof (*y));

    for (int i = 0; i < n; i++)
    {
        y[i] = x[i];
    }

    return y;
}

void dxxy (int n, const double *x, double *y)
{
    /* y <- x .* y */
    int i;
    for (i = 0; i < n; i++)
    {
        y[i] *= x[i];
    }
}

double mean(double *x, int n)
{
    int i;
    double y = 0.;
    double nd = (double) n;

    for (i = 0; i < n; i++)
    {
        y += x[i] / nd;
    }

    return y;
}

void dcopy(int n, const double *x, double *y)
{
    for (int i = 0; i < n; i++)
    {
        y[i] = x[i];
    }
}

void daxpy(int n, double a, const double *x, double *y)
{
    for (int i = 0; i < n; i++) 
    {
        y[i] += a * x[i];
    }
}

int all(int *x, int n)
{
    int tf = 1;

    for (int i = 0; (i < n) & tf; i++)
    {
        tf &= x[i];
    }

    return tf;
}

/*************************************************************************
 * MATRIX FUNCTIONS:
 *   Descriptions list the equivalent MATLAB syntax if applicable
 *   Note: all dense matrices are 2D arrays in column major order, i.e.
 *   A[j] is the j-th column of A
 * **********************************************************************/

cs * spdiags(const double *x, int n)
{
    cs *T, *D; /* T is the workspace, D is the matrix we return */
    int i;
    int *Ti, *Tj;
    double *Tx;

    /* Create triplet matrix, populate it, compress it, free workspace,
     * return */
    T = (cs *)cs_spalloc(n, n, n, 1, 1);

    if (T == NULL) return T;

    Ti = T->i;
    Tj = T->p;
    Tx = T->x;

    for (i = 0; i < n; i++)
    {
       Ti[i] = Tj[i] = i;
       Tx[i] = x[i];
    }

    T->nz = n;
    D = cs_compress(T);
    cs_spfree(T); 

    return D;
}

cs * speye(int n)
{
    /* speye(n, n) */
    cs *eye;
    double *x;

    x = onesv(n);

    if (x == NULL) return NULL;

    eye = spdiags(x, n);
    free(x);

    return (eye);
}

cs * spzeros(int m, int n)
{
    /* spzeros(m, n) */
    cs *Y, *Z;

    Z = (cs *)cs_spalloc(m, n, 0, 1, 1);

    if (Z == NULL) return Z;

    Z->nz = 0;
    Y = cs_compress(Z);
    cs_spfree(Z);

    return Y;
}

cs * horzcat(const cs *A, const cs *B)
{
    /* C = [A B] */
    cs *C, *D;
    int i = 0;
    int j, k;
    int *Ci, *Cj;
    double *Cx;

    /* Check both matrices are in CSC form and have the same number of rows
     * */
    assert((A->nz == -1) & (B->nz == -1) & (A->m == B->m));

    C = (cs *)cs_spalloc(A->m, A->n + B->n, A->nzmax + B->nzmax, 1, 1);

    if (C == NULL) return C;

    Ci = C->i;
    Cj = C->p;
    Cx = C->x;

    for (j = 0; j < A->n; j++) 
    {
        for (k = A->p[j]; k < A->p[j+1]; k++)
        {
            Ci[i] = A->i[k];
            Cj[i] = j;
            Cx[i] = A->x[k];
            i++;
        }
    }

    for (j = 0; j < B->n; j++) 
    {
        for (k = B->p[j]; k < B->p[j+1]; k++)
        {
            Ci[i] = B->i[k];
            Cj[i] = A->n + j;
            Cx[i] = B->x[k];
            i++;
        }
    }

    C->nz = i;
    D = cs_compress(C);
    cs_spfree(C);

    return D;
}

cs * vertcat(const cs *A, const cs *B)
{
    /* C = [A; B] */
    cs *C, *D;
    int i = 0;
    int j, k;
    int *Ci, *Cj;
    double *Cx;

    /* Check both matrices are CSC and have same number of columns */
    assert((A->nz == -1) & (B->nz == -1) & (A->n == B->n));

    C = (cs *)cs_spalloc(A->m + B->m, A->n, A->nzmax + B->nzmax, 1, 1);

    if (C == NULL) return C;

    Ci = C->i;
    Cj = C->p;
    Cx = C->x;

    for (j = 0; j < A->n; j++) 
    {
        for (k = A->p[j]; k < A->p[j+1]; k++)
        {
            Ci[i] = A->i[k];
            Cj[i] = j;
            Cx[i] = A->x[k];
            i++;
        }
    }

    for (j = 0; j < B->n; j++) 
    {
        for (k = B->p[j]; k < B->p[j+1]; k++)
        {
            Ci[i] = B->i[k] + A->m;
            Cj[i] = j;
            Cx[i] = B->x[k];
            i++;
        }
    }

    C->nz = i;
    D = cs_compress(C);
    cs_spfree(C);

    return D;
}

cs * blkdiag(const cs *A, int mb, int nb)
{
    // A . . . .
    // . A . . .
    // . . A . .
    // . . . A .
    // Create block diagonal matrix with A


    cs *B;
    int m = mb * A->m;
    int n = nb * A->n;

    int nblocks = CS_MIN(mb, nb);

    cs *T;
    int *Ti;
    int *Tj;
    double *Tx;

    T = (cs *)cs_spalloc(m, n, nblocks * A->nzmax, 1, 1);

    Ti = T->i;
    Tj = T->p;
    Tx = T->x;

    int i, j, p, k = 0;

    for (int iblk = 0; iblk < nblocks; iblk++)
    {
        for (j = 0; j < A->n; j++)
        {
            for (p = A->p[j]; p < A->p[j+1]; p++)
            {
                i = A->i[p];
                Ti[k] = i + A->m * iblk;
                Tj[k] = j + A->n * iblk;
                Tx[k++] = A->x[p];
            }
        }
    }

    T->nz = k;

    B = cs_compress(T);
    //cs_print(B,0);


    cs_spfree(T);

    return B;
}

cs * matcopy(const cs *A)
{
    cs *C, *Z;
    /* Easy way: copy by adding zero to it */
    Z = spzeros(A->m, A->n);
    C = cs_add(A, Z, 1., 1.);
    cs_spfree(Z);

    return C;
}

cs * subsref(const cs *A, int *Ii, int *Jj, int ni, int nj)
{
    int i, j, k;
    cs *B, *C, *T;
    int *Ti, *Tj; 
    double *Tx;
    cs *Pi, *Pj;

    /* Create projection matrices to remove the rows and columns */

    /* Project rows if necessary */
    if (ni != -1)
    {
        T = (cs *)cs_spalloc(ni, A->m, ni, 1, 1);
        if (T == NULL) return T;

        Ti = T->i;
        Tj = T->p;
        Tx = T->x;
        k = 0;

        for (i = 0; i < ni; i++)
        {
            Ti[k] = i; Tj[k] = Ii[i]; Tx[k++] = 1.0;
        }

        T->nz = k;
        Pi = cs_compress(T);
        cs_spfree(T);

        if (Pi == NULL) return Pi;

        B = cs_multiply(Pi, A);
        cs_spfree(Pi);

        if (B == NULL) return B;
    }
    else
    {
        B = matcopy(A);
    }

    /* Project columns if necessary */
    if (nj != -1)
    {
        T = (cs *)cs_spalloc(A->n, nj, nj, 1, 1);

        if (T == NULL) return T;

        Ti = T->i;
        Tj = T->p;
        Tx = T->x;
        k = 0;

        for (j = 0; j < nj; j++)
        {
            Ti[k] = Jj[j]; Tj[k] = j; Tx[k++] = 1.;
        }

        T->nz = k;
        Pj = cs_compress(T);
        cs_spfree(T);

        if (Pj == NULL) return Pj;

        C = cs_multiply(B, Pj);
        cs_spfree(Pj);
        cs_spfree(B);
    }
    else
    {
        C = B;
    }
    return C;
}

double **sparse2dense(const cs *As)
{
    /* Convert to a dense matrix in COLUMN MAJOR format */
    double **A;
    int i, j, k, m, n;

    m = As->m;
    n = As->n;

    A = (double **)malloc(n * sizeof(*A));

    if (A == NULL) return A;

    /* Allocate the memory for the matrix */
    for (j = 0; j < n; j++)
    {
        A[j] = zerosv(m);

        /* If malloc fails, free all allocated memory and exit */
        if (A[j] == NULL)
        {
            for (i = 0; i < j; i++)
            {
                free(A[i]);
            }

            free(A);

            return NULL;
        }
    }

    /* Memory is successfully allocated at this point */
    for (j = 0; j < n; j++)
    {
        for (k = As->p[j]; k < As->p[j+1]; k++)
        {
            i = As->i[k];
            A[j][i] = As->x[k];
        }
    }

    return A;
}

cs *dense2sparse(double **A, int m, int n)
{
    cs *As, *B;
    int i, j, k = 0;
    int *Ai, *Aj;
    double *Ax;
    As = (cs *)cs_spalloc(m, n, m*n, 1, 1);

    if (As == NULL) return As;

    Ai = As->i;
    Aj = As->p;
    Ax = As->x;

    for (j = 0; j < n; j++)
    {
        for (i = 0; i < m; i++)
        {
            if (A[j][i])
            {
                Ai[k] = i;
                Aj[k] = j;
                Ax[k] = A[j][i];
                k++;
            }
        }
    }

    As->nz = k;
    B = cs_compress(As);
    cs_spfree(As);

    return B;
}

void densefree(double **A, int n)
{
    int j;

    for (j = 0; j < n; j++)
    {
        free(A[j]);
    }

    free(A);
}

void denseprint(double **A, int m, int n)
{
    int i, j;

    for (i = 0; i < m; i++)
    {
        printf("\t");

        for (j = 0; j < n; j++)
        {
            if (A[j][i] == 0)
            {
                printf("     . ");
            }
            else
            {
                printf("%7.2f", A[j][i]);
            }
        }

        printf("\n");
    }
}

void vecprint(double *v, int n)
{
    int i;
    printf("\t");

    for (i = 0; i < n; i++)
    {
        if (v[i] == 0)
        {
            printf("     . ");
        }
        else
        {
            printf("%7.2f", v[i]);
        }
    }

    printf("\n");
}

void sparseprint(cs *A)
{
    double **B;
    B = sparse2dense(A);
    denseprint(B, A->m, A->n);
    densefree(B, A->n);
}

void spy(cs *A)
{
    double **B;
    B = sparse2dense(A);

    for (int i = 0; i < A->m; i++)
    {
        printf("\t");

        for (int j = 0; j < A->n; j++)
        {
            if (B[j][i] == 0)
            {
                printf(". ");
            }
            else
            {
                printf("x ");
            }
        }

        printf("\n");
    }

    densefree(B, A->n);
}

uint32_t imorton_odd(uint32_t x)
{
    /* Compute the morton number corresponding to the odd bits of x */
    x &= 0x55555555;                 // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
    x = (x ^ (x >> 1)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
    x = (x ^ (x >> 2)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
    x = (x ^ (x >> 4)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
    x = (x ^ (x >> 8)) & 0x0000ffff; // x = ---- ---- ---- ---- fedc ba98 7654 3210
    return x;
}

uint32_t imortonx(uint32_t x)
{
    return imorton_odd(x);
}

uint32_t imortony(uint32_t x)
{
    return imorton_odd(x >> 1);
}

void cholsoln(csn *Nu, css *S, int n, double *b, double *x)
{
    cs_ipvec(S->pinv, b, x, n);    /* permute  */
    cs_lsolve(Nu->L, x);           /* lower triangular solve  */
    cs_ltsolve(Nu->L, x);          /* upper triangular solve */
    cs_pvec(S->pinv, x, b, n);     /* inverse permute */
}

cs * mldivide_chol(cs *A, css *S, cs *B)
{
    // Compute A \ B, utilising symbolic Cholesky factorisation in S
     
    // Compute Numerical Cholesky LL^T
    csn *N;
    N = cs_chol(A, S);

    // Do lower triangular solve LY = B
    // Initialise workspace for cs_spsolve
    int n = A->n;
    double *x; int *xi;
    x  = (double *)malloc(n * sizeof (*x));
    xi = (int *)malloc(2*n * sizeof (*xi));

    // Initialise triplet matrix for output 
    cs *T; 
    T = (cs *)cs_spalloc(B->m, B->n, 1, 1, 1); // will be grown with cs_entry

    int nB = B->n;
    int top;

    for (int k = 0; k < nB; k++)
    {
        // k-th column
        top = cs_spsolve(N->L, B, k, xi, x, NULL, 1);
        // Add entries to T
        for (int i = top; i < n; i++)
        {
            cs_entry(T, xi[i], k, x[xi[i]]);
        }
    }

    cs *Y;
    Y = cs_compress(T);
    cs_spfree(T);

    // Now do the upper triangular solve
    T = (cs *)cs_spalloc(Y->m, Y->n, 1, 1, 1); // will be grown with cs_entry

    cs *U;
    U = cs_transpose(N->L, 1);

    for (int k = 0; k < nB; k++)
    {
        // k-th column
        top = cs_spsolve(U, Y, k, xi, x, NULL, 0);

        // Add entries to T
        for (int i = top; i < n; i++)
        {
            cs_entry(T, xi[i], k, x[xi[i]]);
        }
    }

    cs * X;
    X = cs_compress(T);
    cs_spfree(T);
    cs_spfree(U);
    cs_spfree(Y);
    cs_nfree(N);

    free(x);
    free(xi);

    return X;
}

numjac* malloc_numjac()
{
    numjac* N = 0;
    N = (numjac *)malloc(sizeof(*N));
    N->A = 0;
    N->g = 0;
    N->r = 0;
    return N;
}

numjac *numjacinit(cs *A)
{
    // numjacinit computes the column grouping for computing numerical
    // Jacobians using a greedy algorithm, and initialises the sparse matrix in which the 
    // numerical Jacobian will reside
    
    // Initialise data structure
    numjac *N; 
    int m = A->m;
    int n = A->n;
    N = malloc_numjac();
    N->g = (int *)malloc((n+1) * sizeof (*N->g));
    N->r = (int *)malloc((n+1) * sizeof (*N->r)); // Worst case, will be reallocated later

    /* Initialise arrays */
    int *remaining, *excluded;
    remaining = (int *)malloc(n * sizeof (*remaining)); // ungrouped columns
    excluded  = (int *)malloc(m * sizeof (*excluded));   

    /* Initially, all columns are ungrouped */
    for (int j = 0; j < n; j++)
    {
        remaining[j] = j;
    }

    int nrem = n; 

    int irem, j, gnum, ig=0, flag;

    // Loop over number of groups
    for (gnum = 0; nrem > 0; gnum++)
    {
        N->r[gnum] = ig; // Group boundary

        // Initialise excluded array to be none
        for (int i = 0; i < m; i++)
        {
            excluded[i] = 0;
        }

        irem = 0;

        // Iterate over ungrouped columns 
        for (int k = 0; k < nrem; k++)
        {
            j = remaining[k];
            flag = 0;

            // loop over the col of A
            for (int idx = A->p[j]; idx < A->p[j+1]; idx++)
            {
                if (excluded[A->i[idx]])
                {
                    flag = 1; // Row entry already present in excluded
                    break;
                }
            }

            if (flag)   // Column can't be added to group, put in remaining for next group
            {
                remaining[irem++] = j;
            }
            else        // Column is added to group
            {
                N->g[ig++] = j; // add column to group vector

                for (int idx = A->p[j]; idx < A->p[j+1]; idx++)
                {
                    excluded[A->i[idx]] = 1; // add all row indices to excluded
                }
            }
        }

        nrem = irem;
    }

    //printf("%d %d %d\n",gnum,irem,nrem);
    //cs_print(A,1);

    N->r[gnum] = n;
    /*if ((n+1) > (gnum+1))
    {
     N->r = realloc(N->r, (gnum+1)*sizeof (*N->r)); // remove extra entries
    }*/
    N->ng = gnum;
    N->A = matcopy(A); // Create copy of matrix for storing Jacobian
    free(remaining); free(excluded);
    
    return N;
}

// Finds the n-dimensional euclidean distance between two points as n-element shallow arrays.
double distanceBetween2Points(double *a, double *b, const int dimensions)
{   
    double inner = 0;
    for (int i = 0; i < dimensions; i++)
    {
        inner += pow(a[i] - b[i], 2);
    }
    return sqrt(inner);
}
#endif
