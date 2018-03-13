#ifndef NVUNode_H
#define NVUNode_H

#include "Lens.h"
#include "CG_NVUNode.h"
#include "rndm.h"
#include "NVUMacros.h" // nvu index macros
#include <math.h>
#include <cs.h>
#include <stdio.h>
#include <stdlib.h>
#include "matrixOperations.h"
#include "Coordinates.h"
#include "MaxComputeOrder.h"

#define WRITE_STATE_VARS 0
#define WRITE_FLUXES 0
#define PLOTTING_NODE 0

#define LIFE_TIME 100.0 // TODO unit  Stewart?
#define INJECT_TIME 10.0 // TODO unit??? needs to be less than half of (LIFE_TIME - INJECT_TIME)

#define INITIAL_WAIT LIFE_TIME // TODO unit?
#define POW_OF_2(x) (1 << (x)) // macro for 2^x using bitwise shift

#define ITERATIONS_PER_SEC 100000

struct Workspace 
{
    int     nblocks;// Number of nvu blocks
    int     nu;     // Number of equations total
    int     ntimestamps;  //TODO unit Stewart?
    // Jacobian information 
    int    isjac;
    numjac *dfdx;   // derivatives of DEs with respect to state
    cs     *J;

    int fevals;
    int jacupdates;
    double tfeval;
    double tjacupdate;
    double tjacfactorize;

    /* Model-specific stuff */
        // Mandatory fields (accessed externally). Initialised in nvu_init
    int neq;
    int num_fluxes;
    cs *dfdx_pattern; // neq * neq matrix indicating Jacobian structure of nvu 

    // Other NVU parameters for radius and pressure. TODO: rename
    double a1, a2, a3, a4, a5;
    double b1, d1, d2, g1, g2;
    double l;
    double pcap;        // pressure at capillaries (min)
    // from ode_workspace

    csn *N; // Newton matrix numeric factorisation
    css *S; // Newton matrix sybolic factorisation
    double *y; // Workspace variable
    // double *p; //
    // double *q; //
    double *f; // Workspace variable
    double *fluxes;
    //double dt;
    double t0;
    double tf;
    double ftol;
    double ytol;
    int    maxits;
    int    nconv;
    int mdeclared;
    double dtwrite;

    FILE* state_var_file;
    FILE* fluxes_file;

    double *beta;
    double *w;
    double *x;

    int jac_needed;
    int converged;

    double t;
    double total_t;
    double tnext;
    int pressuresIndex;
    int counter;
    double injectStart;

};


class NVUNode : public CG_NVUNode
{
    public:
        void initStateVariables(RNG& rng);
        void initJacobian(RNG& rng);
        void update(RNG& rng);
        void copy(RNG& rng);
        void finalize(RNG& rng);
        virtual ~NVUNode();
    private:
      
    Workspace* workspace;
    void malloc_workspace();
    void newton_matrix();
    int lusoln(double *b);
    css * newton_sparsity(cs *J);
    void back_euler();
    void solver_init();
    void write_state_var_data();
        void write_fluxes_data();


    // From brain.h

    void evaluate(double t, double *y, double *dy);
    void jacupdate(double t, double *y);
    double p0(double t);
    void init_dfdx();

    void eval_dfdx(double t, double *y, double *f, double eps);
    void set_initial_conditions(double *y);

    // from nvu.h
    // Right hand side routine for one block
    void  nvu_rhs(double t, double *u, double *du, double *fluxes);

    //time- and space-dependent Glu input
    double nvu_Glu(double t);

    //time- and space-dependent K+ input
    double K_input(double t);

    //time- and space-dependent flux_ft input
    double flux_ft(double t);

    //time- and space-dependent PLC input
    double PLC_input(double t);

    // ECS K+ input
    double ECS_input(double t);

    //factorial
    double factorial(int c);

    // Initial conditions
    void nvu_ics();
        
    void diffuse();
    void calculate_pressure_index();

    //int sizecheck(double *x, int n, double tol);

};



#endif
