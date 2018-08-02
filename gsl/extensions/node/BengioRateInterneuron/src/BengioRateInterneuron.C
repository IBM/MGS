#include "Lens.h"
#include "BengioRateInterneuron.h"
#include "CG_BengioRateInterneuron.h"
#include "rndm.h"

#define SHD getSharedMembers()

void BengioRateInterneuron::update_U(RNG& rng) 
{
  u += SHD.dT * (-SHD.g_lk * u + SHD.g_D *  (v - u) + i + SHD.sigma * gaussian(rng) );
  phi_u = 1.0/(1.0+exp(-u));

  i = 0;
  ShallowArray<double*>::iterator iter, end=pyramidalTeachingInputs.end();
  for (iter=pyramidalTeachingInputs.begin(); iter!=end; ++iter) {
    double g_Exc = SHD.g_som * (**iter - SHD.E_inh) / (SHD.E_exc - SHD.E_inh);
    double g_Inh = SHD.g_som * (**iter - SHD.E_exc) / (SHD.E_exc - SHD.E_inh);
    i += g_Exc * (SHD.E_exc - u) + g_Inh * (SHD.E_inh - u);
  }  
}

void BengioRateInterneuron::update_V(RNG& rng) 
{
  ShallowArray<Input>::iterator iter, end=pyramidalLateralInputs.end();
  double d_rate =  phi_u - (1.0 / (1.0+exp(-SHD.predictionFactor*v)));
  v = 0;
  for (iter=pyramidalLateralInputs.begin(); iter!=end; ++iter) {
    v += iter->weight * ( *(iter->input) );
    iter->weight += SHD.eta_IP * d_rate * ( *(iter->input) );
  }
}

BengioRateInterneuron::~BengioRateInterneuron() 
{
}

