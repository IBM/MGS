// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Lens.h"
#include "LinskerInfomaxUnit.h"
#include "CG_LinskerInfomaxUnit.h"
#include "GridLayerData.h"
#include "NodeCompCategoryBase.h"
#include "rndm.h"
#include <sstream>
#include <iostream>

#define SHD getSharedMembers()
#define ITER getSimulation().getIteration()

void LinskerInfomaxUnit::initialize(RNG& rng) 
{
  assert(TH.size()>0);
  assert(LN.size()>0);
  assert(SHD.period>1);
  alpha = 1.0;

  ShallowArray<LinskerInfomaxUnit_THinput>::iterator THiter, THend=TH.end();
  ShallowArray<LinskerInfomaxUnit_LNinput>::iterator LNiter, LNend=LN.end();

  double normW=0;
  for (THiter=TH.begin(); THiter!=THend; ++THiter)
    normW+=THiter->weight*THiter->weight;
  assert(normW);
  normW=sqrt(normW);
  for (THiter=TH.begin(); THiter!=THend; ++THiter)
    THiter->weight/=normW;
  // creates self-connections (diagonal lateral connections)
  for (LNiter=LN.begin(); LNiter!=LNend; ++LNiter)
    if (LNiter->e == &ePublic) LNiter->weight=1.0;
}

void LinskerInfomaxUnit::update(RNG& rng) 
{
  if (SHD.inversion_method){
    ShallowArray<LinskerInfomaxUnit_THinput>::iterator THiter, THend=TH.end(); // (possibly whitened) Thalamic inputs
    double u = 0; 
    // FF inputs
    for (THiter=TH.begin(); THiter!=THend; ++THiter){
      u += THiter->weight * *(THiter->x); // u = Cx
    }
    y = 1.0/(1.0+exp(-(u+w0)));
    double antiHebb = (1.0 - 2.0*y);
    for (THiter=TH.begin(); THiter!=THend; ++THiter){
      THiter->deltaW += antiHebb * *(THiter->x);
    }
    deltaW0 += antiHebb;

  } else {
    int phase = ITER % SHD.period;
    ShallowArray<LinskerInfomaxUnit_THinput>::iterator THiter, THend=TH.end(); // Thalamic input from GatedThalamicUnit (FFwd, i.e. matrix C)
    ShallowArray<LinskerInfomaxUnit_LNinput>::iterator LNiter, LNend=LN.end(); // Lateral Network input from other LinskerInfomaxUnits within the same cortical area (i.e. matrix Q)
    double u = 0; 
    double normE=e=0;
     
    // phase to compute antiHebbian term (i.e. the (C')^-1 term in Linsker)
    if (phase==0) {
      double antiHebb = (1.0 - 2.0*y0);
      double Hebb = SHD.betaC * ( ls + antiHebb ); // ls is Psi in Kozloski 2007; Hebb is almost deltaC (but the x' factor is missing)
      // FF inputs
      for (THiter=TH.begin(); THiter!=THend; ++THiter) {
	THiter->weight += Hebb * THiter->xPrev; 
	u += THiter->weight * (THiter->xPrev = *(THiter->x));
      }
      w0 += SHD.betaW0 * antiHebb; // w0 : output bias
      y = y0 = 1.0/(1.0+exp(-(u+w0))); // y : output

      v = u0 = u; 
      // Power iteration method to calculate eigenvalues
      for (LNiter=LN.begin(); LNiter!=LNend; ++LNiter) {
	e += LNiter->weight * *(LNiter->e);   
	normE += *(LNiter->e) * *(LNiter->e);
      }
    }
    // normal phase:
    else {
      // FF inputs
      for (THiter=TH.begin(); THiter!=THend; ++THiter)
	u += THiter->weight * *(THiter->x); // u = Cx
      y = 1.0/(1.0+exp(-(u+w0))); 
      // Power iteration method
      double Qv = 0;
      for (LNiter=LN.begin(); LNiter != LNend; ++LNiter) {
	if (phase==1) LNiter->weight += SHD.betaQ * (v * *(LNiter->v) - LNiter->weight); //deltaQ = betaQ*(v0_i*v0_j' - Q)
	Qv += LNiter->weight * *(LNiter->v); 
	e += LNiter->weight * *(LNiter->e); 
	normE += *(LNiter->e) * *(LNiter->e);
      }
      v += u0 - alpha*Qv; // v(t) = v(t-1) + u - alpha*Q*v(t-1)
      ls = alpha*v; // Psi = alpha*v
    }

    e *= (alpha = 1.0/sqrt(normE));  
  }
}

void LinskerInfomaxUnit::copy(RNG& rng) 
{
  vPublic=v;
  ePublic=e;
}

void LinskerInfomaxUnit::setIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_LinskerInfomaxUnitInAttrPSet* CG_inAttrPset, CG_LinskerInfomaxUnitOutAttrPSet* CG_outAttrPset) 
{
  int row = getGlobalIndex()+1; // +1 is for Matlab
  int col = CG_node->getGlobalIndex()+1;

  if (CG_inAttrPset->identifier=="TH") {
    TH[TH.size()-1].row = row;
    TH[TH.size()-1].col = col;
  }
  else if (CG_inAttrPset->identifier=="LN") {
    LN[LN.size()-1].row = row;
    LN[LN.size()-1].col = col;
  }
  else assert(0);
}

void LinskerInfomaxUnit::outputWeights(std::ofstream& fsTH, std::ofstream& fsLN)
{
  ShallowArray<LinskerInfomaxUnit_THinput>::iterator THiter, 
    THend=TH.end();
  ShallowArray<LinskerInfomaxUnit_LNinput>::iterator LNiter, 
    LNend=LN.end();

  for (THiter=TH.begin(); THiter!=THend; ++THiter)
    fsTH<<THiter->row<<" "<<THiter->col<<" "<<THiter->weight<<std::endl;

  for (LNiter=LN.begin(); LNiter!=LNend; ++LNiter)
    fsLN<<LNiter->row<<" "<<LNiter->col<<" "<<LNiter->weight<<std::endl;
}

void LinskerInfomaxUnit::getInputWeights(std::ofstream& fsW){
  ShallowArray<LinskerInfomaxUnit_THinput>::iterator it;
  ShallowArray<LinskerInfomaxUnit_THinput>::iterator end = TH.end();
  for (it=TH.begin(); it!=end; ++it) {
    fsW << it->weight << " ";
  }
  fsW << std::endl; 
}

void LinskerInfomaxUnit::getInputWeights(std::vector<double>* W_j){
  ShallowArray<LinskerInfomaxUnit_THinput>::iterator it;
  ShallowArray<LinskerInfomaxUnit_THinput>::iterator end = TH.end();
  int idx=0;
  for (it=TH.begin(); it!=end; ++it) {
    (*W_j)[idx] = it->weight;
    idx++;
  }
}

void LinskerInfomaxUnit::setInputWeights(std::vector<double>* newWeights){
  ShallowArray<LinskerInfomaxUnit_THinput>::iterator it;
  ShallowArray<LinskerInfomaxUnit_THinput>::iterator end = TH.end();
  int idx=0;
  for (it=TH.begin(); it!=end; ++it) {
    it->weight += SHD.betaW*(SHD.period * (*newWeights)[idx] + it->deltaW);
    it->deltaW = 0;
    idx++;
  }
  w0 += SHD.betaW*deltaW0;
  deltaW0=0;
}

LinskerInfomaxUnit::~LinskerInfomaxUnit() 
{
}

