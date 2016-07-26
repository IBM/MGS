#include "Lens.h"
#include "ChannelKv31.h"
#include "CG_ChannelKv31.h"
#include "rndm.h"

#include "MaxComputeOrder.h"
#include "GlobalNTSConfig.h"

#define SMALL 1.0E-6
#include <math.h>
#include <pthread.h>
#include <algorithm>

#include "SegmentDescriptor.h"

#if CHANNEL_Kv31 == Kv31_RETTIG_1992
#define IMV -18.7
#define IMD 9.7
#define TMF 4.0
#define TMV 46.56
#define TMD 44.14
#define T_ADJ 2.9529 // 2.3^((34-21)/10)

#else
  NOT DEFINED YET

#endif


void ChannelKv31::update(RNG& rng) 
{
  dyn_var_t dt = *(getSharedMembers().deltaT);
  for (unsigned i = 0; i < branchData->size; ++i)
  {
    dyn_var_t v = (*V)[i];
#if CHANNEL_Kv31 == Kv31_RETTIG_1992
    dyn_var_t minf = 1.0/(1.0 + exp(-(v + IMV)/IMD));
    // dyn_var_taum = TMF/(T_ADJ*(1 + exp(-(v + TMV)/TMD)));
    dyn_var_t taum = TMF/((1 + exp(-(v + TMV)/TMD)) * getSharedMembers().Tadj);
    dyn_var_t pm = 0.5*dt/taum;
    m[i] = (2.0*pm*minf + m[i]*(1.0 - pm))/(1.0 + pm);
#endif
    // trick to keep m in [0, 1]
    if (m[i] < 0.0) { m[i] = 0.0; }
    else if (m[i] > 1.0) { m[i] = 1.0; }
#if CHANNEL_Kv31 == Kv31_RETTIG_1992
    g[i] = gbar[i] * m[i] ;
#endif
		Iion[i] = g[i] * (v - getSharedMembers().E_K[0]);
  }
}

void ChannelKv31::initialize(RNG& rng) 
{
  //pthread_once(&once_KAf, ChannelKAf::initialize_others);
  assert(branchData);
  unsigned size = branchData->size;
  assert(V);
  assert(gbar.size() == size);
  assert(V->size() == size);
  // allocate
  if (g.size() != size) g.increaseSizeTo(size);
  if (m.size() != size) m.increaseSizeTo(size);
  if (Iion.size()!=size) Iion.increaseSizeTo(size);
  // initialize
  float gbar_default = gbar[0];
	if (gbar_dists.size() > 0 and gbar_branchorders.size() > 0)
	{
    std::cerr << "ERROR: Use either gbar_dists or gbar_branchorders on Channels Kv31 Param"
			<< std::endl;
		assert(0);
	}
  for (unsigned i = 0; i < size; ++i)
  {
    if (gbar_dists.size() > 0) {
      unsigned int j;
      //NOTE: 'n' bins are splitted by (n-1) points
      if (gbar_values.size() - 1 != gbar_dists.size())
      {
        std::cerr << "gbar_values.size = " << gbar_values.size()
          << "; gbar_dists.size = " << gbar_dists.size() << std::endl;
      }
      assert(gbar_values.size() -1 == gbar_dists.size());
      for (j=0; j<gbar_dists.size(); ++j) {
        if ((*dimensions)[i]->dist2soma < gbar_dists[j]) break;
      }
      gbar[i] = gbar_values[j];
    } 
		/*else if (gbar_values.size() == 1) {
      gbar[i] = gbar_values[0];
    } */
		else if (gbar_branchorders.size() > 0)
		{
      unsigned int j;
      if (gbar_values.size() != gbar_branchorders.size())
      {
        std::cerr << "gbar_values.size = " << gbar_values.size()
          << "; gbar_branchorders.size = " << gbar_branchorders.size() << std::endl;
      }
      assert(gbar_values.size() == gbar_branchorders.size());
      SegmentDescriptor segmentDescriptor;
      for (j=0; j<gbar_branchorders.size(); ++j) {
        if (segmentDescriptor.getBranchOrder(branchData->key) == gbar_branchorders[j]) break;
      }
			if (j == gbar_branchorders.size() and gbar_branchorders[j-1] == GlobalNTS::anybranch_at_end)
			{
				gbar[i] = gbar_values[j-1];
			}
			else if (j < gbar_values.size()) 
        gbar[i] = gbar_values[j];
      else
        gbar[i] = gbar_default;
		}
		else {
      gbar[i] = gbar_default;
    }
	}
  for (unsigned i = 0; i < size; ++i)
  {
    dyn_var_t v = (*V)[i];
#if CHANNEL_Kv31 == Kv31_RETTIG_1992
    m[i] = 1.0/(1.0 + exp(-(v + IMV)/IMD));
    g[i] = gbar[i]*m[i];
#else
    NOT IMPLEMENTED YET;
// m[i] = am / (am + bm);  // steady-state value
// h[i] = ah / (ah + bh);
#endif

		Iion[i] = g[i] * (v - getSharedMembers().E_K[0]);
 }
}

ChannelKv31::~ChannelKv31() 
{
}

