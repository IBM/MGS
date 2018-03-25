#include "Lens.h"
#include "Htree.h"
#include "CG_Htree.h"
#include "rndm.h"

#define TStep (*(getSharedMembers().deltaT)*1e-3)  //(sec)


const double RMIN  = 10e-6; // meter, radius of smallest vessel (in meter)
const double BIFURCATION_SCALE = 1.4142135623730951; // = sqrt(2), amount the radius decreases by when going down a level
const double LRR   = 20; // unitless - nondimensional, length to radius ratio
const double MU    = 3.5e-3; // Pascal.second, blood viscosity

// TODO these are also in NVU :( Make them extern constants? Or make a static class Constants
const double R0    = 10e-6;  // meter (for nondimensionalising)
const double P0    = 8000.0;  // Pascal (scaling factor for nondim)
const double PCAP  = 4000.0 / P0;  // Pascal (capillary bed pressure), normalized

//#define DEBUG

void Htree::initTree(RNG& rng) 
{
    // Allocate memory for all arrays, structs, and classes inside TreeWorkspace.
    malloc_tree();
    tree->numberNVUs = _numberNVUs; 
    std::cerr<< "Number of NVUs as seen by Htree is " << tree->numberNVUs << std::endl;
    assert(tree->numberNVUs > 0 && "number of NVU's <= 0 (bad input)");

    tree->nu = NEQ * tree->numberNVUs;
    /*
    Parse input parameters and set tree sizes. There are four N values
    that matter: 
        N is the number of levels in the tree (total)
        N0: number of levels in the root subtree
        Np: number of levels in the subtrees corresponding to each
        core (N0 + Np = N)
        Nsub: number of levels in the small scale subtrees for Jacobian
        computation
    */

    tree->N = (2* log2(round(sqrt(tree->numberNVUs)))) + 1;
    tree->Nsub = 1;

    tree->N0 = 0;//(int) round(log2((double) tree->N)); 
    tree->Np = tree->N - tree->N0; // Dependent on number of levels and number of cores running

    init_subtree();                // Init adjacency matrix and workspace for subtree

    // Set IC's for pressure array
    for (int i = 0; i < tree->A->m; i++)
    {
        pressures.push_back(0);
    }
    set_spatial_coordinates();
    compute_symbol_cholesky();

    set_conductance(0);       // set scaled conductances
    set_length();                  // Initialise the vessel lengths

    calculate_indices();

    double r, l;
    int correctedNVUIndex;
    // Update conductances of autoregulating vessels

    updatePressures(rng);

    copy(rng);

    tree->counter = 0;


    tree->QglobalPos = 0;
    tree->PglobalPos = 0;
    init_io();
#if HTREE_IO_CHOICE == IBM_NEW_CODE
    _prevWritingTime = 0.0; // (second)
    write_data(0); 
#elif HTREE_IO_CHOICE == UC_OLD_CODE
    write_info_n_data();
#endif
}

void Htree::updatePressures(RNG& rng) 
{
    double r, l;

    int correctedNVUIndex;
    // Update conductances of autoregulating vessels
    for (int i = 0; i < tree->numberNVUs; i++)
    {
        //r = y[NEQ*i]; // Radius is always the first variable
        correctedNVUIndex = tree->NVUIndices[i];
        r = *(NVUinputs[correctedNVUIndex].radius);
        assert(!isnan(r));
        l = tree->l[i];
        tree->g[i] = pow(r, 4) / l;
    }

    double t = TStep * getSimulation().getIteration();
    double p0 = inputPressure(t);

    compute_uv(PCAP); // compute the intermediate variables u and v

    compute_sub(p0, PCAP); // compute p, w and q  

}

void Htree::finalize(RNG& rng) 
{
    // Write data for the final timestep before quitting
    writeToFiles(rng);
#if HTREE_IO_CHOICE == IBM_NEW_CODE
    write_info();
#elif HTREE_IO_CHOICE == UC_OLD_CODE
#endif

    if (tree != NULL) close_io();

    if (tree->A != NULL) cs_spfree(tree->A);
    if (tree->At != NULL) cs_spfree(tree->At);
    if (tree->G != NULL) cs_spfree(tree->G);
    if (tree->level != NULL) free(tree->level);
    if (tree->symbchol != NULL) cs_sfree(tree->symbchol);
    if (tree->l != NULL) free(tree->l);
    if (tree->b != NULL) free(tree->b);
    if (tree->p != NULL) free(tree->p);
    if (tree->q != NULL) free(tree->q);
    if (tree->w != NULL) free(tree->w);
    if (tree->u != NULL) free(tree->u);
    if (tree->v != NULL) free(tree->v);    
    if (tree->x != NULL) free(tree->x);
    if (tree->y != NULL) free(tree->y);
    if (tree->xm != NULL) free(tree->xm);
    if (tree->stateVars != NULL) free(tree->stateVars);
    if (tree->NVUIndices != NULL) free(tree->NVUIndices);

    if (tree != NULL) free (tree);
}

void Htree::setPointers(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_HtreeInAttrPSet* CG_inAttrPset, CG_HtreeOutAttrPSet* CG_outAttrPset) 
{
    // add connection (array of state variables from a single NVU) to list of all variables
    NVUinputs[NVUinputs.size() - 1].NVUstateVariables = getSharedMembers().tempStateVariables;
    NVUinputs[NVUinputs.size() - 1].NVUcoords = getSharedMembers().tempNVUcoords;
    _numberNVUs += 1; //TODO replace with NVUinputs.size()
}

// Copy pressures from tree->p to pressureArray so that NVUs can read their new pressures
void Htree::copy(RNG& rng)
{
    for (int i = 0; i < tree->A->m; i++) // Number of tree Nodes, NOT number of NVU's
    {
        pressures[i] = tree->p[i];
        
    }
}

#if HTREE_IO_CHOICE == IBM_NEW_CODE
void Htree::writeToFiles(RNG& rng)
{
    double currentTime = TStep * getSimulation().getIteration();
    //if (tree->counter % 100000 == 0)  /* assuming dt=10e-6 second --> output ever 1 second */
    double recordInterval = dtWrite * 1e-3; // (second)
    if (currentTime >= _prevWritingTime+recordInterval)
    {
        _prevWritingTime += recordInterval;

#ifdef DEBUG
        printf("Writing Data at t=%f\n", currentTime);
#endif
        int correctedNVUIndex;

        // Pull out state variables and reorder them before writing them'
        for (int i = 0; i < tree->numberNVUs; i++)
        {
            correctedNVUIndex = tree->NVUIndices[i];
                        
            for (int j = 0; j < NEQ; j++)
            {
                tree->stateVars[i * NEQ + j] = (*NVUinputs[correctedNVUIndex].NVUstateVariables)[j];
            }
        }

        write_data();
        write_flow();

        // Pull out the pressures
        for (int i = 0; i < tree->A->m; i ++)
        {
            tree->p[i] = pressures[i];
        }
        write_pressure();   
        tree->counter++;
    }
}   
#elif HTREE_IO_CHOICE == UC_OLD_CODE
void Htree::writeToFiles(RNG& rng)
{
    if (tree->counter % 100000 == 0) 
    {
        printf("Writing Data at t=%f\n", TStep * getSimulation().getIteration());
        int correctedNVUIndex;

        // Pull out state variables and reorder them before writing them'
        for (int i = 0; i < tree->numberNVUs; i++)
        {
            correctedNVUIndex = tree->NVUIndices[i];
                        
            for (int j = 0; j < NEQ; j++)
            {
                tree->stateVars[i * NEQ + j] = (*NVUinputs[correctedNVUIndex].NVUstateVariables)[j];
            }
        }

        write_data();
        write_flow();

        // Pull out the pressures
        for (int i = 0; i < tree->A->m; i ++)
        {
            tree->p[i] = pressures[i];
        }
        write_pressure();   

        tree->counter = 0;
    }
    tree->counter++;
}   
#endif

Htree::~Htree() 
{
}

void Htree::malloc_tree()
{
    tree = 0;
    tree = (TreeWorkspace *)malloc(sizeof(TreeWorkspace));
    tree->A = 0;
    tree->At = 0;
    tree->G = 0;
    tree->level = 0;
    tree->g = 0;

    tree->symbchol = 0;

    tree->l = 0;
    tree->b = 0;

    tree->p = 0;
    tree->q = 0;
    tree->w = 0;

    tree->u = 0;
    tree->v = 0;
    tree->x = 0;
    tree->y = 0;

    tree->xm= 0;

    tree->Toutfilename = 0;
    tree->Qoutfilename = 0;
    tree->Poutfilename = 0;
    tree->dirName = 0;

    // tree->Toutfile = 0;
    // tree->Qoutfile = 0;
    // tree->Poutfile = 0;
    tree->subarray = 0;
    tree->subarray_single = 0;

    tree->stateVars = 0;
    tree->NVUIndices = 0;
} 

void Htree::init_subtree()
{   
    // First construct the local subtree 
    tree->A    = adjacency(tree->Np); // where Np = N - N0, dependent on the number of levels (N) and cores (2^N0)
    tree->At   = cs_transpose(tree->A, 1);
    tree->G    = speye(tree->A->n);     // creates identity matrix I_n
    tree->g    = tree->G->x;            // numerical values of G
    tree->level = (int *)malloc(tree->A->n * sizeof (*tree->level));

    for (int i = 0; i < tree->A->n; i++)
    {
        tree->level[i] = (int) floor(log2(tree->A->n - i)) + tree->N0;
    }

    tree->numberNVUs = tree->A->m + 1;

    // Initialise workspace variables for solving
    tree->l = (double *) malloc (tree->numberNVUs * (sizeof *tree->l));
    tree->x = (double *)malloc (tree->numberNVUs * (sizeof *tree->x));
    tree->y = (double *)malloc (tree->numberNVUs * (sizeof *tree->y));
    tree->b = (double *)malloc (tree->A->n * (sizeof *tree->b));
    tree->u = (double *)malloc (tree->A->m * (sizeof *tree->u));
    tree->v = (double *)malloc (tree->A->m * (sizeof *tree->v));
    tree->p = (double *)malloc (tree->A->m * (sizeof *tree->p));
    tree->q = (double *)malloc (tree->A->n * (sizeof *tree->q));
    tree->w = (double *)malloc (tree->A->n * (sizeof *tree->w));

    tree->xm = (double *) malloc(tree->A->m * sizeof(*tree->xm));

    tree->stateVars = (double *)malloc (tree->numberNVUs * NEQ * (sizeof *tree->stateVars));
    tree->NVUIndices = (int *) malloc (tree->numberNVUs * (sizeof *tree->NVUIndices));
}

void Htree::set_spatial_coordinates()
{
    // work out some stuff
    //double procs = (double) 1;
    double procs = 1.0;
    int log2P = (int) log2(procs);
    int mlocal, nlocal, mglobal, nglobal;
    int iglobal, jglobal;

    int m, n; // number of rows / cols of blocks globally
    // if rectangular, make it so there are more rows than columns (i.e. if N is even, then N-1 % 2 = 1 and m is bigger)
    m = POW_OF_2((tree->N - 1)/2 + (tree->N - 1) %2);
    n = POW_OF_2((tree->N - 1)/2);

    // Work out arrangement of workers, again, if rectangular, set more rows than columns
    if (tree->N % 2 == 1) // odd number of levels - square
    {
        mglobal = POW_OF_2(log2P / 2);
        nglobal = POW_OF_2( (log2P / 2) + log2P % 2);
    }
    else // even number of levels - rectangular
    {
        mglobal = POW_OF_2( (log2P / 2) + log2P % 2);
        nglobal = POW_OF_2(log2P / 2);
    }
    
    // Work out how many rows / columns of blocks we have for each core
    mlocal = m / mglobal;
    nlocal = n / nglobal;
    int rank = 0;
    iglobal = rank % mglobal;
    jglobal = rank / mglobal;

    double xoffset, yoffset;
    xoffset = (double) ((2*jglobal - (nglobal-1)) * nlocal) * L0;
    yoffset = (double) ((2*iglobal - (mglobal-1)) * mlocal) * L0;

    for (int j = 0; j < nlocal; j++)
    {
        for (int i = 0; i < mlocal; i++)
        {
            tree->x[i + mlocal * j] = xoffset + L0 * (double) (2*j - (nlocal - 1));
            tree->y[i + mlocal * j] = yoffset + L0 * (double) (2*i - (mlocal - 1));
#ifdef DEBUG
        std::cerr<< " tree->x[" << i + mlocal * j << "]" << xoffset + L0 * (double) (2*j - (nlocal - 1)) << "; tree->y " << 
            yoffset + L0 * (double) (2*i - (mlocal - 1)) << 
            std::endl;
#endif
        }
    }

    tree->mlocal = mlocal;
    tree->nlocal = nlocal;
    tree->mglobal = mglobal;
    tree->nglobal = nglobal;
}


void Htree::compute_symbol_cholesky()
{
    cs *X, *Y;
    X = cs_multiply(tree->A, tree->G);
    Y = cs_multiply(X, tree->At);
    cs_spfree(X);

    tree->symbchol = cs_schol(0, Y);  // symbchol = AGAt
    cs_spfree(Y);

    // X = cs_multiply(tree->A0, tree->G0);
    // Y = cs_multiply(X, tree->A0t);
    // cs_spfree(X);

    // tree->symbchol0 = cs_schol(0, Y);    // symbchol0 = AGAt for the root subtree
    // cs_spfree(Y);
}

void Htree::set_conductance(int unscaled)
{
    // if unscaled is true, we can compute the conductances for an unscaled
    // version of the problem.
    double r, l;

    if (unscaled)
    {
        for (int i = 0; i < tree->A->n; i++)
        {
            r = compute_radius(tree->level[i], tree->N);
            l = compute_length(tree->level[i], tree->N);
            tree->g[i] = M_PI * pow(r, 4) / (8.0 * MU * l);
        }
    }
    else
    {
        for (int i = 0; i < tree->A->n; i++)
        {
            r = compute_radius(tree->level[i], tree->N) / R0;
            l = compute_length(tree->level[i], tree->N) / L0;
            tree->g[i] = pow(r, 4) / l;
        }
    }
}

void Htree::compute_uv(double pcap)
{
    // u = -AGb, v = [g,0]
    cs *AG, *B;
    csn *Nu;
    // Set up boundary conditions b - put them in q for now
    for (int i = 0; i < tree->A->m + 1; i++)
    {
        tree->q[i] = -pcap;
    }

    for (int i = tree->A->m + 1; i < tree->A->n; i++)
    {
        tree->q[i] = 0;
    }
    // Solving A*G*At p = -A*G b (b is in q for now)
    // Define the matrices AG = A*G, and B = A*G*At
    AG = cs_multiply(tree->A, tree->G);
    B  = cs_multiply(AG, tree->At);


    // Numerical Cholesky factorisation using precomputed symbolic, symbchol = AGAt. Maybe replaces symbchol with B?
    Nu = cs_chol(B, tree->symbchol);

    if (!Nu) printf("Numerical cholesky decomposition failed in compute_uv");

    cs_spfree(B); // B is (potentially) big, so free it

    // Define the RHS of the u equations
    for (int i = 0; i < tree->A->m; i++)
    {
        tree->u[i] = 0.0;                       /* u = 0 */
    }

    cs_gaxpy(AG, tree->q, tree->u);               /* u = AGb + u = AGb */

    for (int i = 0; i < tree->A->m; i++)
    {
        tree->u[i] = -tree->u[i];                 /* u = -u = -AGb */
    }

    cs_spfree(AG); 
    // And solve, using our computed factorisations - what is this outputting..?
    cholsoln(Nu, tree->symbchol, tree->A->m, tree->u, tree->xm);

    // Define RHS of v equations
    tree->v[tree->A->m-1] = tree->g[tree->A->n-1];

    for (int i = 0; i < tree->A->m - 1; i++)
    {
        tree->v[i] = 0.0;
    }
    cholsoln(Nu, tree->symbchol, tree->A->m, tree->v, tree->xm);
    cs_nfree(Nu); 

}

// The function that actually solves for p, w and q!
void Htree::compute_sub(double p0, double pcap)
{
    double pk;

    pk = p0;
    
    for (int i = 0; i < tree->A->m; i++)
    {
        tree->p[i] = tree->u[i] + pk * tree->v[i];
    }

    // figure out w
    tree->w[tree->A->n - 1] = pk;

    // Set w to boundary conditions b
    for (int i = 0; i < tree->A->m + 1 ; i++)
    {
        tree->w[i] = -pcap;
    }
    for (int i = tree->A->m + 1; i < (tree->A->n - 1); i++)
    {
        tree->w[i] = 0;
    }

    cs_gaxpy(tree->At, tree->p, tree->w); // w = At*p + w (w set to b)

    for (int i = 0; i < tree->A->n; i++)
    {
        tree->q[i] = tree->w[i] * tree->g[i]; // q = w g
    }
}

// Time-varying pressure at the root of the tree. 1 is nominal value. If
// you want to work in unscaled units, make sure you *multiply* by P0
// afterwards
// p0 exists elsewhere... that's no benuo. global constant?
double Htree::inputPressure(double t)
{
    //double p0 = 1. * 8000 / P0; // 8000 Pa   original: 1.5 * 8000 / P0;
    //double p0 = (0.5 * sin(t) + 1) * 8000 / P0; //
    double p0 = 1.5 * 8000.0 / P0;    // no time dependence
    return p0;
}

// ************************ from adjancecy **************************************
/* Structure of the H tree defined by a matrix A consisting of 0, 1 and -1's:
*     The tree is a directed graph with adjacency matrix A
*/

cs * Htree::adjacency(int Np)
{
    /* Size of A matrix dependent on Np: the number of levels of the local subtree
     * which is dependent on the number of cores and number of levels)
    Np = # levels - log2 (# cores), also Np = n_blocks(local) - 1
     */

    // Initialise the sparse matrix for filling
    cs *A, *T;
    int *Ti, *Tj;
    double *Tx;
    int m, n;
    m = POW_OF_2(Np-1) - 1;     // number of rows of T
    n = POW_OF_2(Np) - 1;        // number of cols of T
    T = cs_spalloc(m, n, 3*m, 1, 1); // create T with size m x n
    Ti = T->i;                     // array of ints: row indices (size nzmax = max number of entries)
    Tj = T->p;                     // array of ints: column indices (size nzmax = max number of entries)
    Tx = T->x;                     // array of doubles: numerical values (size nzmax = max number of entries)

    // Set the size of the lowest level grid
    int ncols = POW_OF_2((Np-1)/2);                 // equivalent to 2^((Np-1)/2)
    int nrows = POW_OF_2((Np-1)/2 + (Np-1)%2);
    
    int a, k = 0;
    int k1, k2 = 0;
    int row = 0;
    int col = POW_OF_2(Np-1);
    int xbranch = 0;

    // L loop: from bottom level up to the top of the tree (internal nodes)
    for (int L = Np - 1; L > 0; L--)
    {
        a = POW_OF_2(Np) - POW_OF_2(L+1);
        //b = (1 << N) - (1 << (L  ));
        //c = (1 << N) - (1 << (L-1));

        if (xbranch)
        {
            for (int j = 0; j < ncols; j+=2)
            {
                for (int i = 0; i < nrows; i++)
                {
                    k1 = a + i + j*nrows;
                    k2 = a + i + (j+1)*nrows;
                    Ti[k] = row; Tj[k] = k1; Tx[k++] = 1;
                    Ti[k] = row; Tj[k] = k2; Tx[k++] = 1;
                    Ti[k] = row++; Tj[k] = col++; Tx[k++] = -1;
                }
            }
            ncols /= 2;
        } 
        else
        {
            for (int j = 0; j < ncols; j++)
            {
                for (int i = 0; i < nrows; i+=2)
                {
                    k1 = a + i + j*nrows;
                    k2 = k1 + 1;
                    Ti[k] = row; Tj[k] = k1; Tx[k++] = 1;
                    Ti[k] = row; Tj[k] = k2; Tx[k++] = 1;
                    Ti[k] = row++; Tj[k] = col++; Tx[k++] = -1;
                }
            }
            nrows /= 2;
        }
        xbranch = !xbranch; // switch xbranch 0 <--> 1
    }

    T->nz = k;
    A = cs_compress(T);
    cs_spfree(T);
    //sparseprint(A); // good way to print sparse matrix!
    return A;
}

void Htree::set_length()
{
    // Set lengths of autoregulating vessels
    for (int i = 0; i < tree->numberNVUs; i++)
    {
        tree->l[i] = compute_length(tree->level[i], tree->N) / L0;
    }
}

double Htree::compute_length(int level, int n_levels)
{
    double length;
    length = LRR * RMIN * (double) POW_OF_2((n_levels - level - 1)/ 2);
    return length;
}

double Htree::compute_radius(int level, int n_levels)
{
    double radius;
    //r = RMIN * pow(2., ((double) (n_levels - level - 1)) / 2.);
    radius = RMIN * (double) POW_OF_2((n_levels - level - 1)/ 2);
    return radius;
}


// Writing section/////////////////////////////

void Htree::init_io()
{
    int sizes[2];
    int subsizes[2];
    int starts[2];
    int res=0;
    int i=0;

    // Initialise files for MPI I/O. Requires init_parallel to have been
    // called first

    char tSuffix[] = "/tissueBlocks.dat";
    char qSuffix[] = "/flow.dat";
    char pSuffix[] = "/pressure.dat";

    tree->dirName = (char*)malloc(FILENAMESIZE/2 * sizeof(*tree->dirName));

    sprintf(tree->dirName, "np%02d_nlev%02d_sbtr%02d", 1, tree->N, tree->Nsub);

    res = mkdir(tree->dirName, S_IRWXU | S_IRWXG | S_IRWXO); //res == 0 then the directory doesn't exist

    //if dirName already exists, add a suffix (eg. _1, _2 etc)
    while (res == -1)
    {
        i++;
        sprintf(tree->dirName, "np%02d_nlev%02d_sbtr%02d_%d", 1, tree->N, tree->Nsub, i);
        res = mkdir(tree->dirName, S_IRWXU | S_IRWXG | S_IRWXO);
    }


    tree->Toutfilename = (char*) malloc(FILENAMESIZE * sizeof(*tree->Toutfilename));
    sprintf(tree->Toutfilename, "%s%s", tree->dirName, tSuffix);
    dataOut = new std::ofstream(tree->Toutfilename, std::ios::binary);

    //MPI_File_open(MPI_COMM_WORLD, tree->Toutfilename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &tree->Toutfile);

    tree->Qoutfilename = (char*)malloc(FILENAMESIZE*sizeof(*tree->Qoutfilename));
    sprintf(tree->Qoutfilename, "%s%s", tree->dirName, qSuffix);
    flowOut = new std::ofstream(tree->Qoutfilename, std::ios::binary);
    // //MPI_File_open(MPI_COMM_WORLD, tree->Qoutfilename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &tree->Qoutfile);

    tree->Poutfilename = (char*)malloc(FILENAMESIZE*sizeof(*tree->Poutfilename));
    sprintf(tree->Poutfilename, "%s%s",tree->dirName,pSuffix);
    pressureOut = new std::ofstream(tree->Poutfilename, std::ios::binary);
}


void Htree::close_io()
{
    delete dataOut;
    delete flowOut;
    delete pressureOut;

    free(tree->Toutfilename);
    free(tree->Qoutfilename);
    free(tree->Poutfilename);
}

#if HTREE_IO_CHOICE == IBM_NEW_CODE
void Htree::write_data(int part)
{
    if (part == 0)
    {
        std::ofstream& output=*dataOut;

        for (int i = 0; i < tree->numberNVUs; i++)
        {
            output.write(reinterpret_cast<const char*> (&tree->x[i]), sizeof(tree->x[i]));
        }
        for (int i = 0; i < tree->numberNVUs; i++)
        {
            output.write(reinterpret_cast<const char*> (&tree->y[i]), sizeof(tree->y[i]));
        }
    }
    else{
        double t = TStep * getSimulation().getIteration();
        std::ofstream& output=*dataOut;

        output.write(reinterpret_cast<const char*> (&t), sizeof(t));

        for (int i = 0; i < tree->nu; i++)
        {
            output.write(reinterpret_cast<const char*> (&(tree->stateVars[i])), sizeof(tree->stateVars[i]));
        }
    }
}
#elif HTREE_IO_CHOICE == UC_OLD_CODE
void Htree::write_data()
{
    double t = TStep * getSimulation().getIteration();
    std::ofstream& output=*dataOut;

    output.write(reinterpret_cast<const char*> (&t), sizeof(t));

    for (int i = 0; i < tree->nu; i++)
    {
        output.write(reinterpret_cast<const char*> (&(tree->stateVars[i])), sizeof(tree->stateVars[i]));
    }
}
#endif

void Htree::write_flow()
{
    double t = TStep * getSimulation().getIteration();
    std::ofstream& output=*flowOut;

    output.write(reinterpret_cast<const char*> (&t), sizeof(t));

    for (int i = 0; i < tree->A->n; i++)
    {
        output.write(reinterpret_cast<const char*> (&tree->q[i]), sizeof(tree->q[i]));
    }
}

void Htree::write_pressure()
{
    double t = TStep * getSimulation().getIteration();
    std::ofstream& output=*pressureOut;
    output.write(reinterpret_cast<const char*> (&t), sizeof(t));

    for (int i = 0; i < tree->A->m; i++)
    {
        output.write(reinterpret_cast<const char*> (&tree->p[i]), sizeof(tree->p[i]));
    }
}

#if HTREE_IO_CHOICE == IBM_NEW_CODE
/* write some reference data for reading back data 
 *   8 fields
 * HISTORY:
 *    2018-03-24:
 *        write at the end of the simulation as some new data are unknown until the end
 *        add 'n_datapoints'
 *        move piece of the code writing true data to the associated location in write_data(0)
 */
void Htree::write_info()
{
    // Write the summary info to disk
    char * infofilename;

    FILE *fp;
    // Write the data file
    char iSuffix[] = "/info.dat";
    infofilename = (char *)malloc(FILENAMESIZE * sizeof(*infofilename));
    sprintf(infofilename, "%s%s", tree->dirName, iSuffix);

    fp = fopen(infofilename, "w");
    fprintf(fp, "n_processors    n_blocks_per_rank        n_state_vars   m_local         n_local         m_global        n_global     n_datapoints\n");
    fprintf(fp, "%-16d", 1);
    fprintf(fp, "%-16d", tree->numberNVUs);
    fprintf(fp, "%-16d", NEQ);
    fprintf(fp, "%-16d", tree->mlocal);
    fprintf(fp, "%-16d", tree->nlocal);
    fprintf(fp, "%-16d", tree->mglobal);
    fprintf(fp, "%-16d", tree->nglobal);
    fprintf(fp, "%-16d", tree->counter);
    fprintf(fp, "\n");
    fclose(fp);

    free(infofilename);
}
#elif HTREE_IO_CHOICE == UC_OLD_CODE
void Htree::write_info_n_data()
{
    // Write the summary info to disk
    char * infofilename;

    FILE *fp;
    // Write the data file
    char iSuffix[] = "/info.dat";
    infofilename = (char *)malloc(FILENAMESIZE * sizeof(*infofilename));
    sprintf(infofilename, "%s%s", tree->dirName, iSuffix);

    fp = fopen(infofilename, "w");
    fprintf(fp, "n_processors    n_blocks_per_rank        n_state_vars   m_local         n_local         m_global        n_global\n");
    fprintf(fp, "%-16d", 1);
    fprintf(fp, "%-16d", tree->numberNVUs);
    fprintf(fp, "%-16d", NEQ);
    fprintf(fp, "%-16d", tree->mlocal);
    fprintf(fp, "%-16d", tree->nlocal);
    fprintf(fp, "%-16d", tree->mglobal);
    fprintf(fp, "%-16d", tree->nglobal);
    fprintf(fp, "\n");
    fclose(fp);

    free(infofilename);

    std::ofstream& output=*dataOut;

    for (int i = 0; i < tree->numberNVUs; i++)
    {
        output.write(reinterpret_cast<const char*> (&tree->x[i]), sizeof(tree->x[i]));
    }
    for (int i = 0; i < tree->numberNVUs; i++)
    {
        output.write(reinterpret_cast<const char*> (&tree->y[i]), sizeof(tree->y[i]));
    }
}
#endif

/** For each item in the coordinates from NVU nodes, find their index in my x,y coordinates.

*/
void Htree::calculate_indices()
{
    // put x and y arrays into coordinates array to use here and for NVU nodes to compare with on their next phase
    //todo think about removing x,y arrays and only use mdl level shallowarray. wirint to mpi with strides?

    for (int i = 0; i < tree->numberNVUs; i++)
    {
        for (int j = 0; j < DIMENSIONS; j++)
        {
            /* NOTE: as NVU can be on a different process of the Htree
             * NVU's coord from other ranks are only available since InitPhase 
             */
            coordsArray.push_back((*NVUinputs[i].NVUcoords)[j]);
        }
#ifdef DEBUG
        std::cerr<< " nvu " << i << " has coordsArray: " << 
            (*NVUinputs[i].NVUcoords)[0] <<","<<  
            (*NVUinputs[i].NVUcoords)[1] <<","<<
            (*NVUinputs[i].NVUcoords)[2] <<
            std::endl;
#endif
    }

    NVUinput currentNVU;

    double *currentNVUCoords = (double *)malloc(DIMENSIONS * sizeof(double*));
    double *treeCoords = (double *)malloc(DIMENSIONS * sizeof(double*));

    double distance;
    double shortestDistance;
    int currentIndex;
    int bestIndex;

    // LOOP OVER NVU GRID COORDINATES
    for (int i = 0; i < tree->numberNVUs; i++)
    {
        // Assign the current NVU we are searching for in the H tree's coordinates array.
        currentNVU = NVUinputs[i];

        for (int j = 0; j < DIMENSIONS; j++)
        {
            currentNVUCoords[j] = (*currentNVU.NVUcoords)[j];
        }

        // Get first coordinate from tree array to initalise shortest distance.
        for (int j = 0; j < DIMENSIONS; j++)
        {
            treeCoords[j] = coordsArray[j];
        }
        shortestDistance = distanceBetween2Points(currentNVUCoords, treeCoords, DIMENSIONS);

        currentIndex = 0;
        bestIndex = 0;

        // LOOP OVER H TREE COORDINATES
        // search for currentNVU in coordsArray. Begin at second set of coordinates.
        for (int k = DIMENSIONS; k < tree->numberNVUs * DIMENSIONS; k += DIMENSIONS)
        {
            // Increment first because we've already looked at the first item.
            currentIndex++;

            // Fill treeCoords
            for (int j = 0; j < DIMENSIONS; j++)
            {
                treeCoords[j] = coordsArray[k + j];
            }
            // Find the distance between these two coordinates,
            // and update distance if they are closer than the previous best.
            distance = distanceBetween2Points(currentNVUCoords, treeCoords, DIMENSIONS);
            if (distance < shortestDistance)
            {
                shortestDistance = distance;
                bestIndex = currentIndex;
            }
        }

        // Record the best index for one NVU node.
        tree->NVUIndices[i] = bestIndex;
    }

    free(currentNVUCoords);
    free(treeCoords);

    std::cout << "H-tree: Finished calculating tree indices\n";
}
