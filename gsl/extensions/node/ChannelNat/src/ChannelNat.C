#include "Lens.h"
#include "ChannelNat.h"
#include "CG_ChannelNat.h"
#include "rndm.h"

#include "../../nti/include/MaxComputeOrder.h"
#define SMALL 1.0E-6


// 
// This is an implementation of the "Fast, inactivating Na^+ current, I_Nat".
//
//a_m  = AMC*(V + AMV)/( exp( (V + AMV)/AMD ) - 1.0 )
//b_m  = BMC * exp( (V + BMV)/BMD )
//a_h  = AHC * exp( (V + AHV)/AHD )
//b_h  = BHC / (exp( (V + BHV)/BHD ) + 1.0)
#ifdef NAT_HODGKINHUXLEY_1952
#define AMC -0.1
#define AMV 40.0
#define AMD -10.0
#define BMC 4.0
#define BMV 65.0
#define BMD -18.0
#define AHC 0.07
#define AHV 65.0
#define AHD -20.0
#define BHC 1.0
#define BHV 35.0
#define BHD 10.0
#endif
#ifdef NAT_WOLF_2005
//nucleus accumbens (from hippocampal pyramidal cell)
//COMMENT
//Martina M, Jonas P (1997). "Functional differences in na+ channel gating between fast-
//spiking interneurons and principal neurons of rat hippocampus." J Phys, 505(3): 593-603.
//
//recorded at 22C - Q10 of 3 to convert to 35C
//
//
#define AMC -0.1
#define AMV 23.9
#define AMD -11.8
#define BMC 4.0
#define BMV 65.0
#define BMD -18.0
#define AHC 0.07
#define AHV 65.0
#define AHD -20.0
#define BHC 1.0
#define BHV 35.0
#define BHD 10.0
#endif

dyn_var_t ChannelHayNat::vtrap(dyn_var_t x, dyn_var_t y) {
  return(fabs(x/y) < SMALL ? y*(1 - x/y/2) : x/(exp(x/y) - 1));
}

void ChannelNat::update(RNG& rng) 
{
}

void ChannelNat::initialize(RNG& rng) 
{
#ifdef DEBUG_ASSERT
  assert(branchData);
#endif
  unsigned size=branchData->size;
#ifdef DEBUG_ASSERT
  assert(V);
  assert(gbar.size()==size);
  assert(V->size()==size);
#endif
  // allocate 
  if (g.size()!=size) g.increaseSizeTo(size);
  if (m.size()!=size) m.increaseSizeTo(size);
  if (h.size()!=size) h.increaseSizeTo(size);
  // initialize
  for (unsigned i=0; i<size; ++i) {
    gbar[i]=gbar[0];
    dyn_var_t v=(*V)[i];
    dyn_var_t am = AMC*vtrap(v + AMV, AMD);
    dyn_var_t bm = BMC*vtrap(v + BMV, BMD);
    m[i] = am/(am + bm);
    dyn_var_t ah = AHC*vtrap(v + AHV, AHD);
#ifdef NAT_HODGKINHUXLEY_1952
    dyn_var_t bh = BHC/(1.0 + exp((v + BHV)/BHD));
#endif
    h[i] = ah/(ah + bh);
    g[i]=gbar[i]*m[i]*m[i]*m[i]*h[i];
  }
}

ChannelNat::~ChannelNat() 
{
}

