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
#include "NaChannel.h"
#include "CG_NaChannel.h"
#include "rndm.h"

#define SMALL 1.0E-6

#define HH_1952
//#define SCHWEIGHOFER_1999

#ifdef HH_1952
#define AMC 0.1
#define AMV 40.0
#define AMD 10.0
#define BMC 4.0
#define BMV 65.0
#define BMD 18.0
#define AHC 0.07
#define AHV 65.0
#define AHD 20.0
#define BHC 1.0
#define BHV 35.0
#define BHD 10.0
#endif
#ifdef SCHWEIGHOFER_1999
#define AMC 0.1
#define AMV 41.0
#define AMD 10.0
#define BMC 9.0
#define BMV 66.0
#define BMD 20.0
// This is 5/170 to account for the 170 in \tau_h
#define AHC 0.029411764705882
#define AHV 60.0
#define AHD 15.0
// This is 1/170 to account for the 170 in \tau_h
#define BHC 0.005882352941176
#define BHV 50.0
#define BHD 10.0
#endif 

float NaChannel::vtrap(float x, float y) {
  return(fabs(x/y) < SMALL ? y*(1 - x/y/2) : x/(exp(x/y) - 1));
}

void NaChannel::update(RNG& rng)
{
  float dt = *(getSharedMembers().deltaT);
  for (unsigned i=0; i<branchData->size; ++i) {
    float v=(*V)[i];
    float am = AMC*vtrap(-(v + AMV), AMD);
    float bm = BMC*exp(-(v + BMV)/BMD);
    float ah = AHC*exp(-(v + AHV)/AHD);
#ifdef HH_1952
    float bh = BHC/(1.0 + exp(-(v + BHV)/BHD));
#endif
#ifdef SCHWEIGHOFER_1999
    float bh = BHC*vtrap(-(v + BHV), BHD);
#endif

#ifdef HH_1952
    float pm = 0.5*dt*(am + bm);
    m[i] = (dt*am + m[i]*(1.0 - pm))/(1.0 + pm);
#endif
#ifdef SCHWEIGHOFER_1999
    m[i] = am/(am + bm);
#endif
    float ph = 0.5*dt*(ah + bh);
    h[i] = (dt*ah + h[i]*(1.0 - ph))/(1.0 + ph);
    g[i] = gbar[i]*m[i]*m[i]*m[i]*h[i];
  }
}

void NaChannel::initializeNaChannels(RNG& rng)
{
  assert(branchData);
  unsigned size=branchData->size;
  assert(V);
  assert(gbar.size()==size);
  assert(V->size()==size);
  if (g.size()!=size) g.increaseSizeTo(size);
  if (m.size()!=size) m.increaseSizeTo(size);
  if (h.size()!=size) h.increaseSizeTo(size);
  for (unsigned i=0; i<size; ++i) {
    gbar[i]=gbar[0];
    float v=(*V)[i];
    float am = AMC*vtrap(-(v + AMV), AMD);
    float bm = BMC*exp(-(v + BMV)/BMD);
    float ah = AHC*exp(-(v + AHV)/AHD);
#ifdef HH_1952
    float bh = BHC/(1.0 + exp(-(v + BHV)/BHD));
#endif
#ifdef SCHWEIGHOFER_1999
    float bh = BHC*vtrap(-(v + BHV), BHD);
#endif
    m[i] = am/(am + bm);
    h[i] = ah/(ah + bh);
    g[i]=gbar[i]*m[i]*m[i]*m[i]*h[i];
  }
}

NaChannel::~NaChannel()
{
}

