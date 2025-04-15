// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Lens.h"
#include "ZhengSORNInhUnit.h"
#include "CG_ZhengSORNInhUnit.h"
#include "rndm.h"
#include <fstream>
#include <sstream>
#include <iostream>

#define SHD getSharedMembers()
#define ITER getSimulation().getIteration()

void ZhengSORNInhUnit::initialize(RNG& rng) 
{
  spike=false;
  ya = drandom(rng) * ( (drandom(rng)<0.2) ? 1.0 : 0.0 );
  TI = drandom(rng)*SHD.TI_max;

  double sumE=0;
  ShallowArray<SpikeInput>::iterator iter, end=lateralExcInputs.end();
  for (iter=lateralExcInputs.begin(); iter!=end; ++iter) sumE+=iter->weight;
  for (iter=lateralExcInputs.begin(); iter!=end; ++iter) {
	if (sumE!=0) (iter->weight)/=sumE;
  }	
}

void ZhengSORNInhUnit::update(RNG& rng) 
{
  double sumE = 0;
  ShallowArray<SpikeInput>::iterator iter, end=lateralExcInputs.end();
  for (iter=lateralExcInputs.begin(); iter!=end; ++iter)
    if (*(iter->spike)) sumE += iter->weight;
  y = sumE - TI + SHD.sigma2_chi*gaussian(rng);
}

void ZhengSORNInhUnit::fire(RNG& rng) 
{
  spikePrev=spike;
  spike=(y>0);
  ya = SHD.tau_STDP * ( (y>0) ? 1.0 : 0.0 );
}

void ZhengSORNInhUnit::setIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_ZhengSORNInhUnitInAttrPSet* CG_inAttrPset, CG_ZhengSORNInhUnitOutAttrPSet* CG_outAttrPset) 
{
  lateralExcInputs[lateralExcInputs.size()-1].row = getGlobalIndex()+1; // +1 is for Matlab
  lateralExcInputs[lateralExcInputs.size()-1].col = CG_node->getGlobalIndex()+1;
}

void ZhengSORNInhUnit::outputWeights(std::ofstream& fsE2I)
{
  ShallowArray<SpikeInput>::iterator iter, end=lateralExcInputs.end();
  for (iter=lateralExcInputs.begin(); iter!=end; ++iter) {
    fsE2I<<iter->row<<" "<<iter->col<<" "<<iter->weight<<std::endl;
  }
}

void ZhengSORNInhUnit::inputWeights(std::ifstream& fsE2I, int col, float weight)
{
  ShallowArray<SpikeInput>::iterator E2Iiter, E2Iend=lateralExcInputs.end();
  for (E2Iiter=lateralExcInputs.begin(); E2Iiter!=E2Iend; ++E2Iiter) {
    if (E2Iiter->col==col) {
      E2Iiter->weight = static_cast<double>(weight);
      break;
    }
  }
}

void ZhengSORNInhUnit::inputTI(float val)
{
  TI=val; 
}

ZhengSORNInhUnit::~ZhengSORNInhUnit() 
{
}

