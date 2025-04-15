// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Lens.h"
#include "ChannelHayKp.h"
#include "CG_ChannelHayKp.h"
#include "rndm.h"
#include "SegmentDescriptor.h"
#include "Branch.h"
#include "GlobalNTSConfig.h"
#include "NumberUtils.h"

#define SMALL 1.0E-6

#define IMV 11.0
#define IMD 12.0
#define IHV 64.0
#define IHD 11.0
#define TMC 1.25
#define TMF1 175.03
#define TMF2 13.0
#define TMV 10.0
#define TMD1 0.026
#define TMD2 -0.026
#define THC1 360.0
#define THC2 1010.0
#define THF 24.0
#define THV1 65.0
#define THV2 85.0
#define THD 48.0
#define T_ADJ 2.9529 // 2.3^((34-21)/10)


void ChannelHayKp::update(RNG& rng)
{
  float dt = *(getSharedMembers().deltaT);
  for (unsigned i = 0; i<branchData->size; ++i) {
    float v = (*V)[i];
    float minf = 1.0/(1.0 + exp(-(v + IMV)/IMD));
    float taum;
    if (v < -60.0) {
      taum = (TMC + TMF1*exp(TMD1*(v + TMV)))/T_ADJ;
    } else {
      taum = (TMC + TMF2*exp(TMD2*(v + TMV)))/T_ADJ;
    }
    float hinf = 1.0/(1.0 + exp((v + IHV)/IHD));
    float tauh = (THC1 + (THC2 + THF*(v + THV1))*exp(-pow((v + THV2)/THD,2)))/T_ADJ;
    float pm = 0.5*dt/taum;
    float ph = 0.5*dt/tauh;
    m[i] = (2.0*pm*minf + m[i]*(1.0 - pm))/(1.0 + pm);
    h[i] = (2.0*ph*hinf + h[i]*(1.0 - ph))/(1.0 + ph);
    g[i] = gbar[i]*m[i]*m[i]*h[i];
  }
}

void ChannelHayKp::initialize(RNG& rng)
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
    g[i] = gbar[i]*m[i]*m[i]*h[i];
  }
}

ChannelHayKp::~ChannelHayKp()
{
}

