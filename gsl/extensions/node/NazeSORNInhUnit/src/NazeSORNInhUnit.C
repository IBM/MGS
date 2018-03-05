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
#include "NazeSORNInhUnit.h"
#include "CG_NazeSORNInhUnit.h"
#include "rndm.h"
#include <fstream>
#include <sstream>
#include <iostream>

#define SHD getSharedMembers()
#define ITER getSimulation().getIteration()

void NazeSORNInhUnit::initialize(RNG& rng) 
{
  spike=false;
  ya = drandom(rng) * ( (drandom(rng)<0.2) ? 1.0 : 0.0 );
  HIP = std::abs(SHD.mu_HIP+gaussian(rng)*SHD.sigma_HIP);
  double  median = std::log(SHD.mu_IP);
  double sigma = std::sqrt(std::log( std::pow(SHD.sigma_IP,2) / std::pow(SHD.mu_IP,2) + 1));
  eta_IP = exp(median + sigma*std::abs(gaussian(rng)));
  TI = drandom(rng)*SHD.TI_max;

  // Setup buffer containing the delayed input
  int di = std::rand() % SHD.mu_delay.size();
  sigma_delay = SHD.mu_delay[di]*SHD.ratio_delay;
  if (std::round(SHD.mu_delay[di]) > 0) {
    ShallowArray<NazeSORNDelayedSynapseInput>::iterator it_exc, end_exc = lateralExcInputs.end();
    for(it_exc=lateralExcInputs.begin(); it_exc!=end_exc; it_exc++){    
      it_exc->delay = std::lrint(std::abs(SHD.mu_delay[di] + gaussian(rng)*sigma_delay));
      if (it_exc->delay<1) it_exc->delay=1;
      it_exc->spikes_circ_buffer.increaseSizeTo(it_exc->delay + 1);  //!\ assumes delay given in dt units !
      it_exc->a_circ_buffer.increaseSizeTo(it_exc->delay + 2);  //!\ +2 because stores a and aPrev
    }
  }

  // Normalize Exc2Inh weights
  double sumE=0;
  ShallowArray<NazeSORNDelayedSynapseInput>::iterator iter, end=lateralExcInputs.end();
  for (iter=lateralExcInputs.begin(); iter!=end; ++iter) sumE+=iter->weight;
  for (iter=lateralExcInputs.begin(); iter!=end; ++iter) {
	if (sumE!=0) (iter->weight) /= (sumE/SHD.EIratio);
  }	
  
  // Normalize Inh2Inh weights
  double sumI=0;
  ShallowArray<SpikeInput>::iterator iter2, end2=lateralInhInputs.end();
  for (iter2=lateralInhInputs.begin(); iter2!=end2; ++iter2) sumI+=iter2->weight;
  for (iter2=lateralInhInputs.begin(); iter2!=end2; ++iter2) {
	if (sumI!=0) (iter2->weight) /= sumI;
  }	

}

void NazeSORNInhUnit::update(RNG& rng) 
{
  double sumE = 0;
  ShallowArray<NazeSORNDelayedSynapseInput>::iterator iter, end=lateralExcInputs.end();
  for (iter=lateralExcInputs.begin(); iter!=end; ++iter)
    if (*(iter->spike)) sumE += iter->weight;
  
  double sumI = 0;
  ShallowArray<SpikeInput>::iterator iter2, end2=lateralInhInputs.end();
  for (iter2=lateralInhInputs.begin(); iter2!=end2; ++iter2)
    if (*(iter2->spike)) sumI += iter2->weight;

  TI = TI + eta_IP*(spike-HIP);
  y = sumE - sumI - TI + SHD.sigma2_chi*gaussian(rng);
}

void NazeSORNInhUnit::fire(RNG& rng) 
{
  spikePrev=spike;
  spike=(y>0);
  yaPrev = ya;
  ya = SHD.tau_STDP * ya + ( (y>0) ? 1.0 : 0.0 );
  ya = (ya>1.0) ? 1.0 : ya;
}

void NazeSORNInhUnit::setExcIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_NazeSORNInhUnitInAttrPSet* CG_inAttrPset, CG_NazeSORNInhUnitOutAttrPSet* CG_outAttrPset) 
{
  lateralExcInputs[lateralExcInputs.size()-1].row = getGlobalIndex()+1; // +1 is for Matlab
  lateralExcInputs[lateralExcInputs.size()-1].col = CG_node->getGlobalIndex()+1;
}

void NazeSORNInhUnit::setInhIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_NazeSORNInhUnitInAttrPSet* CG_inAttrPset, CG_NazeSORNInhUnitOutAttrPSet* CG_outAttrPset) 
{
  lateralInhInputs[lateralInhInputs.size()-1].row = getGlobalIndex()+1; // +1 is for Matlab
  lateralInhInputs[lateralInhInputs.size()-1].col = CG_node->getGlobalIndex()+1;
}

void NazeSORNInhUnit::outputWeights(std::ofstream& fsE2I)
{
  ShallowArray<NazeSORNDelayedSynapseInput>::iterator iter, end=lateralExcInputs.end();
  for (iter=lateralExcInputs.begin(); iter!=end; ++iter) {
    fsE2I<<iter->row<<" "<<iter->col<<" "<<iter->weight<<std::endl;
  }
}

void NazeSORNInhUnit::inputWeights(std::ifstream& fsE2I, int col, float weight)
{
<<<<<<< HEAD
  ShallowArray<NazeSORNDelayedSynapseInput>::iterator E2Iiter, E2Iend=lateralExcInputs.end();
=======
  ShallowArray<SpikeInput>::iterator E2Iiter, E2Iend=lateralExcInputs.end();
>>>>>>> New model NazeSORNUnit based on ZhengSORNUnit with new features (recurrent inh, adaptive inh thresholds, new data collectors, paramSpace scripts and analysis, etc..)
  for (E2Iiter=lateralExcInputs.begin(); E2Iiter!=E2Iend; ++E2Iiter) {
    if (E2Iiter->col==col) {
      E2Iiter->weight = static_cast<double>(weight);
      break;
    }
  }
}

void NazeSORNInhUnit::inputTI(float val)
{
  TI=val; 
}

NazeSORNInhUnit::~NazeSORNInhUnit() 
{
}

