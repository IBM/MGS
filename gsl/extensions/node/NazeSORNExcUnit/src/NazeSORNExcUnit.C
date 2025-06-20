// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Mgs.h"
#include "NazeSORNExcUnit.h"
#include "CG_NazeSORNExcUnit.h"
#include "GridLayerData.h"
#include "NodeCompCategoryBase.h"
#include "rndm.h"
#include <sstream>
#include <iostream>

#define SHD getSharedMembers()
#define ITER getSimulation().getIteration()

void NazeSORNExcUnit::initialize(RNG& rng) 
{
  spike=drandom(rng)>0.8;
  spikePrev=drandom(rng)>0.8;
  a = drandom(rng) * ( (drandom(rng)>0.8) ? 1.0 : 0.0 );
  aPrev = 0.0;
  HIP = std::abs(SHD.mu_HIP+gaussian(rng)*SHD.sigma_HIP);
  double  median = std::log(SHD.mu_IP);
  double sigma = std::sqrt(std::log( std::pow(SHD.sigma_IP,2) / std::pow(SHD.mu_IP,2) + 1));
  eta_IP = exp(median + sigma*std::abs(gaussian(rng)));
  TE = drandom(rng)*SHD.TE_max;
  //std::cout << "median = " << median << ";  sigma = " << sigma <<";  etaIP = " << eta_IP << ";  HIP = " << HIP << std::endl;
 
  // Get delay index randomly 
  //std::random_device rd;
  //std::mt19937 eng(rd);
  //std::uniform_int_distribution<> distr(0, SHD.mu_delay.size()-1);
  int di = std::rand() % SHD.mu_delay.size();
  sigma_delay = SHD.mu_delay[di]*SHD.ratio_delay;
  // Setup buffer containing the delayed input
  if (std::round(SHD.mu_delay[di]) > 0) {
    ShallowArray<NazeSORNDelayedSynapseInput>::iterator it_exc, end_exc = lateralExcInputs.end();
    for(it_exc=lateralExcInputs.begin(); it_exc!=end_exc; it_exc++){    
      it_exc->delay = std::lrint(std::abs(SHD.mu_delay[di] + gaussian(rng)*sigma_delay));
      if (it_exc->delay<1) it_exc->delay=1;
      it_exc->spikes_circ_buffer.increaseSizeTo(it_exc->delay + 1);  //!\ assumes delay given in dt units !
      it_exc->a_circ_buffer.increaseSizeTo(it_exc->delay + 2);  //!\ +2 because stores a and aPrev
    }
  }
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

void NazeSORNExcUnit::update(RNG& rng) 
{
  // delay, normalization and sum of L5 Exc inputs
  double sumW=0;
  double sumE=0;
  ShallowArray<NazeSORNDelayedSynapseInput>::iterator iter, end=lateralExcInputs.end();
  for (iter=lateralExcInputs.begin(); iter!=end; ++iter) 
    if (iter->synapse) {
      sumW += iter->weight;
      if(iter->delay > 0) {
        iter->spikes_circ_buffer[ITER % iter->delay] = *(iter->spike);
        iter->a_circ_buffer[ITER % iter->delay] = *(iter->a);
      }
  }
  for (iter=lateralExcInputs.begin(); iter!=end; ++iter) {
    if (iter->synapse && sumW!=0) {
      (iter->weight) /= (sumW/SHD.EIratio);
      if (iter->delay>0) {
        if (iter->spikes_circ_buffer[(ITER+1) % iter->delay]) sumE += iter->weight;
      } 
      else if (*(iter->spike)) sumE += iter->weight;
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
  //Normalization of I2E (not in original SORN)
  double sumI = 0;
  sumW = 0;
  ShallowArray<SORNSynapseInput>::iterator iter2, end2=lateralInhInputs.end();
  for (iter2=lateralInhInputs.begin(); iter2!=end2; ++iter2) 
    if (iter2->synapse) sumW += iter2->weight;
  for (iter2=lateralInhInputs.begin(); iter2!=end2; ++iter2) {
    if (iter2->synapse && sumW!=0) {
      (iter2->weight)/=sumW;	
      if (*(iter2->spike)) sumI += iter2->weight;
    }
  }  
  // Sum of L5 Inh inputs
  /*ShallowArray<SORNSynapseInput>::iterator iter2, end2=lateralInhInputs.end();
  for (iter2=lateralInhInputs.begin(); iter2!=end2; ++iter2)
    if (iter2->synapse && *(iter2->spike)) sumI += iter2->weight;*/
  
  // integration of inputs
  double stim = 0;
  if (tmsInput.input) stim=*(tmsInput.input);
  x = sumE - sumI - TE + SHD.sigma2_chi*gaussian(rng) + stim;
  double newSpike = (x>0) ? 1.0 : 0.0;
  // update threshold
  TE = TE + eta_IP*(spike-HIP);
  // update exc synapses (E2E)
  for (iter=lateralExcInputs.begin(); iter!=end; ++iter) {
    if (iter->synapse) {
      // NEW STDP RULE:
      //iter->weight += SHD.eta_STDP * ( spike * *(iter->spikePrev) - spikePrev * *(iter->spike) ); 
      //iter->weight += SHD.eta_STDP * ( a * *(iter->aPrev) - aPrev * *(iter->a) ); 
      iter->weight += SHD.eta_STDP * ( a * iter->a_circ_buffer[(ITER+1) % iter->delay] \
					- aPrev * iter->a_circ_buffer[(ITER+2) % iter->delay] ); // a stored at mod ITER+2; aPrev at %ITER+1
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
      	  //iter2->weight -= (spike ? SHD.eta_iLTP : SHD.eta_inhib);  // eta_iLTP=-0.01; eta_inhib=0.001
      	  iter2->weight -= SHD.eta_inhib * (1.0 - newSpike*SHD.eta_iSTDP) * *(iter2->a);
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

void NazeSORNExcUnit::fire(RNG& rng) 
{
  // Spike update
  spikePrev=spike;
  spike=(x>0);
  // STDP accumulator update
  aPrev = a;
  a = SHD.tau_STDP*a + ((x>0) ? 1.0 : 0.0);
  a = (a>1.0) ? 1.0 : a;
}

void NazeSORNExcUnit::checkForSynapse(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_NazeSORNExcUnitInAttrPSet* CG_inAttrPset, CG_NazeSORNExcUnitOutAttrPSet* CG_outAttrPset) 
{
  if (CG_inAttrPset->condition<CG_inAttrPset->conditionalFraction) {
    lateralExcInputs[lateralExcInputs.size()-1].synapse=true;
  }
  else lateralExcInputs[lateralExcInputs.size()-1].weight=0.0;
}

void NazeSORNExcUnit::checkForInhSynapse(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_NazeSORNExcUnitInAttrPSet* CG_inAttrPset, CG_NazeSORNExcUnitOutAttrPSet* CG_outAttrPset) 
{
  if (CG_inAttrPset->condition<CG_inAttrPset->conditionalFraction) {
    lateralInhInputs[lateralInhInputs.size()-1].synapse=true;
  }
  else lateralInhInputs[lateralInhInputs.size()-1].weight=0.0;
}

void NazeSORNExcUnit::setE2EIndices(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_NazeSORNExcUnitInAttrPSet* CG_inAttrPset, CG_NazeSORNExcUnitOutAttrPSet* CG_outAttrPset) 
{
  lateralExcInputs[lateralExcInputs.size()-1].row = getGlobalIndex()+1; // +1 is for Matlab
  lateralExcInputs[lateralExcInputs.size()-1].col = CG_node->getGlobalIndex()+1;
}

void NazeSORNExcUnit::setI2EIndices(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_NazeSORNExcUnitInAttrPSet* CG_inAttrPset, CG_NazeSORNExcUnitOutAttrPSet* CG_outAttrPset) 
{
  lateralInhInputs[lateralInhInputs.size()-1].row = getGlobalIndex()+1; // +1 is for Matlab
  lateralInhInputs[lateralInhInputs.size()-1].col = CG_node->getGlobalIndex()+1;
}

bool NazeSORNExcUnit::checkInitWeights(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_NazeSORNExcUnitInAttrPSet* CG_inAttrPset, CG_NazeSORNExcUnitOutAttrPSet* CG_outAttrPset) 
{
  if (SHD.initWeights.size()==0) return true;
  else {
    int outIdx = getNode()->getIndex()+1;		//+1 because compatibility w/ matlab
    int inIdx = CG_node->getNode()->getIndex()+1;
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

void NazeSORNExcUnit::outputWeights(std::ofstream& fsE2E, std::ofstream& fsI2E)
{
  ShallowArray<NazeSORNDelayedSynapseInput>::iterator iter, end=lateralExcInputs.end();
  for (iter=lateralExcInputs.begin(); iter!=end; ++iter) {
    if(iter->synapse) fsE2E<<iter->row<<" "<<iter->col<<" "<<iter->weight<<std::endl;
  }
  
  ShallowArray<SORNSynapseInput>::iterator iter2, end2=lateralInhInputs.end();
  for (iter2=lateralInhInputs.begin(); iter2!=end2; ++iter2) {
    if(iter2->synapse) fsI2E<<iter2->row<<" "<<iter2->col<<" "<<iter2->weight<<std::endl;
  }
}

void NazeSORNExcUnit::outputDelays(std::ofstream& fsE2Ed)
{
  ShallowArray<NazeSORNDelayedSynapseInput>::iterator iter, end=lateralExcInputs.end();
  for (iter=lateralExcInputs.begin(); iter!=end; ++iter) {
    if(iter->synapse) fsE2Ed<<iter->row<<" "<<iter->col<<" "<<iter->delay<<std::endl;
  }
}

void NazeSORNExcUnit::inputWeights(int col, float weight)
{
  ShallowArray<NazeSORNDelayedSynapseInput>::iterator E2Eiter, E2Eend=lateralExcInputs.end();
  for (E2Eiter=lateralExcInputs.begin(); E2Eiter!=E2Eend; ++E2Eiter) {
    if (E2Eiter->col==col) {
      E2Eiter->synapse=true;
      E2Eiter->weight = static_cast<double>(weight);
      break; 
    }
  }
}

void NazeSORNExcUnit::inputI2EWeights(int col, float weight)
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

void NazeSORNExcUnit::inputTE(float val)
{
  TE=val; 
}

void NazeSORNExcUnit::getInitParams(std::ofstream& fs_etaIP, std::ofstream& fs_HIP)
{
  fs_etaIP << eta_IP << " ";
  fs_HIP << HIP << " ";
}

NazeSORNExcUnit::~NazeSORNExcUnit() 
{
}

