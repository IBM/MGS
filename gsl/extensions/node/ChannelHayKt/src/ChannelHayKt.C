// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "Lens.h"
#include "ChannelHayKt.h"
#include "CG_ChannelHayKt.h"
#include "rndm.h"
#include "SegmentDescriptor.h"
#include "Branch.h"
#include "GlobalNTSConfig.h"

#define SMALL 1.0E-6

#define IMV 10.0
#define IMD 19.0
#define IHV 76.0
#define IHD 10.0
#define TMC 0.34
#define TMF 0.92
#define TMV 81.0
#define TMD 59.0
#define THC 8.0
#define THF 49.0
#define THV 83.0
#define THD 23.0
#define T_ADJ 2.9529 // 2.3^((34-21)/10)

float ChannelHayKt::vtrap(float x, float y) {
	return(fabs(x/y) < SMALL ? y*(1 - x/y/2) : x/(exp(x/y) - 1));
}

void ChannelHayKt::update(RNG& rng)
{
  float dt = *(getSharedMembers().deltaT);
  for (unsigned i = 0; i<branchData->size; ++i) {
    float v = (*V)[i];
    float minf = 1.0/(1.0 + exp(-(v + IMV)/IMD));
    float taum = (TMC + TMF*exp(-pow((v + TMV)/TMD,2)))/T_ADJ;
    float hinf = 1.0/(1.0 + exp((v + IHV)/IHD));
    float tauh = (THC + THF*exp(-pow((v + THV)/THD,2)))/T_ADJ;
    float pm = 0.5*dt/taum;
    float ph = 0.5*dt/tauh;
    m[i] = (2.0*pm*minf + m[i]*(1.0 - pm))/(1.0 + pm);
    h[i] = (2.0*ph*hinf + h[i]*(1.0 - ph))/(1.0 + ph);
    g[i] = gbar[i]*m[i]*m[i]*m[i]*m[i]*h[i];
  }
}

void ChannelHayKt::initialize(RNG& rng)
{
  unsigned size=branchData->size;
  assert(V);
  assert(gbar.size()==size);
  assert(V->size()==size);
  if (g.size()!=size) g.increaseSizeTo(size);
  if (m.size()!=size) m.increaseSizeTo(size);
  if (h.size()!=size) h.increaseSizeTo(size);
  float gbar_default = gbar[0];
	if (gbar_dists.size() > 0 and gbar_branchorders.size() > 0)
	{
    std::cerr << "ERROR: Use either gbar_dists or gbar_branchorders on Channels Param"
			<< std::endl;
		assert(0);
	}
  for (int i = 0; i<size; ++i) {
		//gbar init
		if (gbar_dists.size() > 0) {
			unsigned int j;
			assert(gbar_values.size() == gbar_dists.size() + 1);
			for (j=0; j<gbar_dists.size(); ++j) {
				if ((*dimensions)[i]->dist2soma < gbar_dists[j]) break;
			}
			if (j < gbar_values.size()) 
				gbar[i] = gbar_values[j];
			else
				gbar[i] = gbar_default;
		} 
		/*else if (gbar_values.size() == 1) {
			gbar[i] = gbar_values[0];
			} */
		else if (gbar_branchorders.size() > 0)
		{
			unsigned int j;
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
  for (unsigned i=0; i<size; ++i) {
    float v = (*V)[i];
    m[i] = 1.0/(1.0 + exp(-(v + IMV)/IMD));
    h[i] = 1.0/(1.0 + exp((v + IHV)/IHD));
    g[i] = gbar[i]*m[i]*m[i]*m[i]*m[i]*h[i];
  }
}

ChannelHayKt::~ChannelHayKt()
{
}

