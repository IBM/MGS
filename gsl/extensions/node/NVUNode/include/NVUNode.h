// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef NVUNode_H
#define NVUNode_H

#include "Mgs.h"
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

#define OPTION_LIFE 0
#define OPTION_NEURON 1
#define INPUT_TO_NVU OPTION_LIFE
//#define INPUT_TO_NVU OPTION_NEURON 


//I/O or not, and if so, on what node?
#define WRITE_STATE_VARS 1
#define WRITE_FLUXES 1
//what NVUNode to plot
#define PLOTTING_NODE 0

#define LIFE_TIME 100.0 // (sec) 
#define INJECT_TIME 10.0 // (sec) needs to be less than half of (LIFE_TIME - INJECT_TIME)

#define INITIAL_WAIT LIFE_TIME // (sec) 
#define POW_OF_2(x) (1 << (x)) // macro for 2^x using bitwise shift

#define ITERATIONS_PER_SEC 100000

//#define MODEL_DIFFUSE
#define SIMULATE_ISOLATE_VESSEL_CLAMP_PRESSURE

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

#if INPUT_TO_NVU == OPTION_LIFE
    //time- and space-dependent Glu input
    double nvu_Glu(double t);
    //time- and space-dependent K+ input
    double K_input(double t);  
#elif INPUT_TO_NVU == OPTION_NEURON
    void nvu_Glu();
    void nvu_K();
#endif

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
    void add_or_update_extra_stateVariables(int condition); //0 = add, 1 = update (default)

    //int sizecheck(double *x, int n, double tol);
    int _gIdx;
    //float _idxCounter;

    double Glut_out; //uM  - glut-concentration (in MegaSC) as a result of release/reuptake from presynaptic(firing) neuron only [NOT counting remval effect from other cells]
    //double K_out; //mM  - Potassium-concentration as a result of ....
    double J_Na_out; // uM/sec - flux of uptake Na+ from MegaSynapticCleft into (presynaptic/firing) neuron
    double J_K_out; // uM/sec  - flux of release K+ into MegaSynapticCleft
    double J_Glut_out; // uM/sec - flux of release Glut into MegaSynapticCleft
    /* NOTE: K+ in the extracellular space (EC) is calculated in ODE */

};



#endif
