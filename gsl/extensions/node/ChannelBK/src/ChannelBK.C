#include "Lens.h"
#include "ChannelBK.h"
#include "CG_ChannelBK.h"
#include "rndm.h"

#define SMALL 1.0E-6

#include <math.h>
#include <pthread.h>
#include <algorithm>

#define Cai_base 0.1e-6 // [uM]
//
// Implementation of the KCa potassium current
//  Voltage and Calcium dependent Potassium
#if CHANNEL_BK == BK_TRAUB_1994
// This implementation is from Traub et al. 1994
//  CA1 pyramidal neuron model
//  Used in Traub's models of several neuron types
//  Non-inactivating, V- and Ca-dep K
//  Alpha-Beta forward-backward rate formulation model
#else
    NOT IMPLEMENTED YET
#endif

void ChannelBK::update(RNG& rng) 
{
    dyn_var_t dt = *(getSharedMembers().deltaT);
    for (unsigned i = 0; i < branchData->size; ++i)
    {
        dyn_var_t v = (*V)[i];
#if SIMULATION_INVOLVE == VMONLY
        dyn_var_t cai = Cai_base;            
#else
        dyn_var_t cai = (*Cai)[i]; // [uM]
#endif

#if CHANNEL_BK == BK_TRAUB_1994
        dyn_var_t alpha;
        if ( v <= 50 ) {
            alpha = exp(((v-10)/11) - ((v-6.5)/27))/18.975;
        } else if ( v > 50) {
            alpha = 2*exp(-((v-6.5)/27));
        }
        dyn_var_t beta = 2*exp(-((v-6.5)/27))-alpha;
        // Rempe * Chopp (2006)
        dyn_var_t pc = 0.5*dt*(alpha+beta);
        fO[i] = (dt*alpha + fO[i]*(1.0-pc))/(1.0+pc);
        dyn_var_t CaGate = (cai/250.0)>1.0?1.0:(cai/250.0);
        g[i] = gbar[i]*fO[i]*CaGate;
    }
#endif
}

void ChannelBK::initialize(RNG& rng) 
{
    unsigned size=branchData->size;
    assert(V);
    assert(gbar.size()==size);
    assert (V->size()==size);
    if (fO.size()!=size) fO.increaseSizeTo(size);
    for (unsigned i = 0; i < size; ++i) {
#if SIMULATION_INVOLVE == VMONLY
        dyn_var_t cai = Cai_base;            
#else
        dyn_var_t cai = (*Cai)[i]; // [uM]
#endif
        gbar[i]=gbar[0];
        dyn_var_t alpha ;
        dyn_var_t v=(*V)[i];
        if ( v <= 50 ) {
            alpha = exp(((v-10)/11) - ((v-6.5)/27))/18.975;
        } else if ( v > 50) {
            alpha = 2*exp(-((v-6.5)/27));
        }
        dyn_var_t beta = 2*exp(-((v-6.5)/27))-alpha;
        fO[i] = alpha/(alpha+beta); // steady-state value
        dyn_var_t CaGate = (cai/250.0)>1.0?1.0:(cai/250.0);
        g[i] = gbar[i]*fO[i]*CaGate;
    }
}

ChannelBK::~ChannelBK() 
{
}

