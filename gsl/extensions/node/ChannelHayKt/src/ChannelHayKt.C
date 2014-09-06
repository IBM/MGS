// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
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
  for (int i = 0; i<size; ++i) {
    gbar[i] = gbar[0];
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

