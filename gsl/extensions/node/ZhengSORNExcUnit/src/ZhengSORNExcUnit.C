// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2006-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "Lens.h"
#include "ZhengSORNExcUnit.h"
#include "CG_ZhengSORNExcUnit.h"
#include "GridLayerData.h"
#include "NodeCompCategoryBase.h"
#include "rndm.h"
#include <sstream>
#include <iostream>

#define SHD getSharedMembers()
#define ITER getSimulation().getIteration()

void ZhengSORNExcUnit::initialize(RNG& rng) 
{
  spike=drandom(rng)>0.8;
  spikePrev=drandom(rng)>0.8;
  a = drandom(rng) * ( (drandom(rng)>0.8) ? 1.0 : 0.0 );
  aPrev = 0.0;
  HIP = SHD.mu_HIP+gaussian(rng)*SHD.sigma_HIP;
  if (HIP<0) HIP=0;
  double  median = std::log(SHD.mu_IP);
  double sigma = std::sqrt(std::log( std::pow(SHD.sigma_IP,2) / std::pow(SHD.mu_IP,2) + 1));
  eta_IP = exp(median + sigma*std::abs(gaussian(rng)));
  TE = drandom(rng)*SHD.TE_max;
  // Normalization of I2E synaptic weights at initialization only
  double sumI=0;
  ShallowArray<SORNSynapseInput>::iterator iter, end=lateralInhInputs.end();
  for (iter=lateralInhInputs.begin(); iter!=end; ++iter) {
    if(iter->synapse) sumI+=iter->weight;
  }
  for (iter=lateralInhInputs.begin(); iter!=end; ++iter) {
    if (sumI!=0 && iter->synapse) (iter->weight)/=sumI;	
  }
  
  if (normalizedThInput.input){
    normalizedThInput.minVal = 0.0;
    normalizedThInput.maxVal = 1.0;
  }
}

void ZhengSORNExcUnit::update(RNG& rng) 
{
  // normalization and sum of L5 Exc inputs
  double sumW=0;
  double sumE=0;
  ShallowArray<SORNSynapseInput>::iterator iter, end=lateralExcInputs.end();
  for (iter=lateralExcInputs.begin(); iter!=end; ++iter) 
    if (iter->synapse) sumW += iter->weight;
  for (iter=lateralExcInputs.begin(); iter!=end; ++iter) {
    if (iter->synapse && sumW!=0) {
      (iter->weight)/=sumW;	
      if (*(iter->spike)) sumE += iter->weight;
    }
  }
  // modulation from L2/3 or Thalamus
  if (modulatoryInput.input && ITER > SHD.Ach_time) { 
    sumE *= (1-SHD.Ach*(1-*(modulatoryInput.input)))/(1.0-SHD.Ach/2.0);
  } else {
    if (normalizedThInput.input && ITER > SHD.Ach_time) {
      if(*(normalizedThInput.input) < normalizedThInput.minVal) normalizedThInput.minVal = *(normalizedThInput.input);
      if(*(normalizedThInput.input) > normalizedThInput.maxVal) normalizedThInput.maxVal = *(normalizedThInput.input);
      sumE *= (1-SHD.Ach*(1-((*(normalizedThInput.input)-normalizedThInput.minVal)/normalizedThInput.maxVal)))/(1.0-SHD.Ach/2.0);
    }
  }
  // Sum of L5 Inh inputs
  //sumW = 0;
  double sumI = 0;
  /*ShallowArray<SynapseInput>::iterator iter2, end2=lateralInhInputs.end();
  for (iter2=lateralInhInputs.begin(); iter2!=end2; ++iter2) 
    if (iter2->synapse) sumW += iter2->weight;
  for (iter2=lateralInhInputs.begin(); iter2!=end2; ++iter2) {
    if (iter2->synapse && sumW!=0) {
      (iter2->weight)/=sumW;	
      if (*(iter2->spike)) sumI += iter2->weight;
    }
  }*/ //Normalization not needed for I2E at the moment
  ShallowArray<SORNSynapseInput>::iterator iter2, end2=lateralInhInputs.end();
  for (iter2=lateralInhInputs.begin(); iter2!=end2; ++iter2)
    if (iter2->synapse && *(iter2->spike)) sumI += iter2->weight; 
  
  // integration of inputs
  x = sumE - sumI - TE + SHD.sigma2_chi*gaussian(rng);
  double newSpike = (x>0) ? 1.0 : 0.0;
  // update threshold
  TE = TE + eta_IP*(spike-HIP);
  // update exc synapses (E2E)
  for (iter=lateralExcInputs.begin(); iter!=end; ++iter) {
    if (iter->synapse) {
      iter->weight += SHD.eta_STDP * ( spike * *(iter->spikePrev) - spikePrev * *(iter->spike) ); 
      // NEW STDP RULE: (used with accu and tau_STDP)
      //iter->weight += SHD.eta_STDP * ( a * *(iter->aPrev) - aPrev * *(iter->a) ); 
      // pruning
      if (iter->weight<0.0001) {
        iter->weight=0;
        iter->synapse=false;
      }
    } 
    else {
      // create new synapses with connection probability p_c (structural plasticity)
      // /!\ todo: Should be only between existing areas (see A_0 in matlab code) 
      if (drandom(rng)<SHD.p_c) {
        iter->synapse = true;
	iter->weight = 0.001;
      }
    }
  }
  // update inh synapses (I2E)
  for (iter2=lateralInhInputs.begin(); iter2!=end2; ++iter2) {
    if(iter2->synapse){
	if (*(iter2->spikePrev)) {
      	  iter2->weight -= (spike ? SHD.eta_iLTP : SHD.eta_inhib);  // eta_iLTP=-0.01; eta_inhib=0.001
          // NEW STDP RULE: (used with accu and tau_STDP)
      	  //iter2->weight -= SHD.eta_inhib * (1.0 - newSpike*SHD.eta_iSTDP) * *(iter2->a);
        }
        if (iter2->weight<0.001) {
	  iter2->weight=0.001;
	  //iter2->synapse=false;
	}
    } /* else { 
	// structural plasticity
    	if (drandom(rng)<SHD.p_c){
	  iter2->weight=0.001;
	  iter2->synapse=true;
    	}
      }*/
  }
}

void ZhengSORNExcUnit::fire(RNG& rng) 
{
  // Spike update
  spikePrev=spike;
  spike=(x>0);
  // STDP accumulator update
  aPrev = a;
  a = SHD.tau_STDP*a + ((x>0) ? 1.0 : 0.0);
  a = (a>1.0) ? 1.0 : a;
}

void ZhengSORNExcUnit::checkForSynapse(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_ZhengSORNExcUnitInAttrPSet* CG_inAttrPset, CG_ZhengSORNExcUnitOutAttrPSet* CG_outAttrPset) 
{
  if (CG_inAttrPset->condition<CG_inAttrPset->conditionalFraction) {
    lateralExcInputs[lateralExcInputs.size()-1].synapse=true;
  }
  else lateralExcInputs[lateralExcInputs.size()-1].weight=0.0;
}

void ZhengSORNExcUnit::checkForInhSynapse(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_ZhengSORNExcUnitInAttrPSet* CG_inAttrPset, CG_ZhengSORNExcUnitOutAttrPSet* CG_outAttrPset) 
{
  if (CG_inAttrPset->condition<CG_inAttrPset->conditionalFraction) {
    lateralInhInputs[lateralInhInputs.size()-1].synapse=true;
  }
  else lateralInhInputs[lateralInhInputs.size()-1].weight=0.0;
}

void ZhengSORNExcUnit::setE2EIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_ZhengSORNExcUnitInAttrPSet* CG_inAttrPset, CG_ZhengSORNExcUnitOutAttrPSet* CG_outAttrPset) 
{
  lateralExcInputs[lateralExcInputs.size()-1].row = getGlobalIndex()+1; // +1 is for Matlab
  lateralExcInputs[lateralExcInputs.size()-1].col = CG_node->getGlobalIndex()+1;
}

void ZhengSORNExcUnit::setI2EIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_ZhengSORNExcUnitInAttrPSet* CG_inAttrPset, CG_ZhengSORNExcUnitOutAttrPSet* CG_outAttrPset) 
{
  lateralInhInputs[lateralInhInputs.size()-1].row = getGlobalIndex()+1; // +1 is for Matlab
  lateralInhInputs[lateralInhInputs.size()-1].col = CG_node->getGlobalIndex()+1;
}

bool ZhengSORNExcUnit::checkInitWeights(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_ZhengSORNExcUnitInAttrPSet* CG_inAttrPset, CG_ZhengSORNExcUnitOutAttrPSet* CG_outAttrPset) 
{
  if (SHD.initWeights.size()==0) return true;
  else {
    int outIdx = getNode()->getIndex();
    int inIdx = CG_node->getNode()->getIndex();
    ShallowArray<float>::const_iterator it = SHD.initWeights.begin();
    ShallowArray<float>::const_iterator end = SHD.initWeights.end();
    for (it; it<end; it+=3) {
      if(outIdx == static_cast<int>(*it)) {   				//row
	if(inIdx == static_cast<int>(*(it+1))) {  			//col
	  CG_inAttrPset->weight = static_cast<double>(*(it+2));		//val
	  std::cout << outIdx << " " << inIdx << " " << CG_inAttrPset->weight << std::endl;
	  return true;
	}
      }
    }
    return false;
  }
}

void ZhengSORNExcUnit::outputWeights(std::ofstream& fsE2E, std::ofstream& fsI2E)
{
  ShallowArray<SORNSynapseInput>::iterator iter, end=lateralExcInputs.end();
  for (iter=lateralExcInputs.begin(); iter!=end; ++iter) {
    if(iter->synapse) fsE2E<<iter->row<<" "<<iter->col<<" "<<iter->weight<<std::endl;
  }
  
  ShallowArray<SORNSynapseInput>::iterator iter2, end2=lateralInhInputs.end();
  for (iter2=lateralInhInputs.begin(); iter2!=end2; ++iter2) {
    if(iter2->synapse) fsI2E<<iter2->row<<" "<<iter2->col<<" "<<iter2->weight<<std::endl;
  }
}

void ZhengSORNExcUnit::inputWeights(int col, float weight)
{
  ShallowArray<SORNSynapseInput>::iterator E2Eiter, E2Eend=lateralExcInputs.end();
  for (E2Eiter=lateralExcInputs.begin(); E2Eiter!=E2Eend; ++E2Eiter) {
    if (E2Eiter->col==col) {
      E2Eiter->synapse=true;
      E2Eiter->weight = static_cast<double>(weight);
      break; 
    }
  }
}

void ZhengSORNExcUnit::inputI2EWeights(int col, float weight)
{
  ShallowArray<SORNSynapseInput>::iterator I2Eiter, I2Eend=lateralInhInputs.end();
  for (I2Eiter=lateralInhInputs.begin(); I2Eiter!=I2Eend; ++I2Eiter) {
    if (I2Eiter->col==col) {
      I2Eiter->synapse=true;
      I2Eiter->weight = static_cast<double>(weight);
      break; 
    }
  }
}

void ZhengSORNExcUnit::inputTE(float val)
{
  TE=val; 
}

ZhengSORNExcUnit::~ZhengSORNExcUnit() 
{
}

