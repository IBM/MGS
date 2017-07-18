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
#include "RabinovichWinnerlessUnit.h"
#include "CG_RabinovichWinnerlessUnit.h"
#include "GridLayerData.h"
#include "rndm.h"
#include <map>
#include <fstream>
#include <sstream>
#include <stdio.h>

#define SHD getSharedMembers()
#define ITER getSimulation().getIteration()

void RabinovichWinnerlessUnit::initialize(RNG& rng) 
{
  if (SHD.Bstim)
    {
      stim.increaseSizeTo(SHD.Nstim);
      for (int c=0; c<SHD.Nstim; ++c)
        stim[c]=SHD.phi*gaussian(rng);
    }
  
  burst = (x>0) ? x : 0.0;

  if (SHD.corticostriatalPlasticity)
    {
      if (D==1)
        GABAt = SHD.GABAtD1;
      else if (D==2)
        GABAt = SHD.GABAtD2;
      else
        assert(0);
  
      // shuffle the dopamine synapses across the cortical inputs
      if (dopamineInputs.size()>0) {
        ShallowArray<ModulatedSynapseInput>::iterator
          iter1=corticalInputs.begin(), end1=corticalInputs.end();
        while (iter1!=end1) {
          std::map<double, SpikeInput*> randomizer;
          ShallowArray<SpikeInput>::iterator iter3, end3=dopamineInputs.end();
          for (iter3=dopamineInputs.begin(); iter3!=end3; ++iter3)
            randomizer[drandom(rng)]=&(*iter3);
          std::map<double, SpikeInput*>::iterator
            iter4=randomizer.begin(), end4=randomizer.end();
          while (iter1!=end1 && iter4!=end4) {
            iter1->modulator=iter4->second;
            ++iter1;
            ++iter4;
          }
        }
      }
    }
}

void RabinovichWinnerlessUnit::update(RNG& rng) 
{
  // Cortico-striatal inputs
  double drive=0.0;
  ShallowArray<ModulatedSynapseInput>::iterator iter1, end1=corticalInputs.end();
  for (iter1=corticalInputs.begin(); iter1!=end1; ++iter1)
    if (iter1->synapse && *(iter1->spike) )
      drive += iter1->weight;

  // Unit's plasticity trace
  if (SHD.lateralPlasticity)
    P0=(P0+burst)*SHD.tauP;
  
  // Striatal lateral inputs
  double IN=0.0;
  //ShallowArray<PlasticInput>::iterator iter2, end2=lateralInputs.end();
  ShallowArray<StructuralInput>::iterator iter2, end2=lateralInputs.end();
  for (iter2=lateralInputs.begin(); iter2!=end2; ++iter2) {
    if (iter2->synapse)
      IN += *(iter2->input) * iter2->weight;
    // TODO: needed for plasticity?
    if (SHD.lateralPlasticity)
      {
        /*    if (SHD.eta_inhib>0) {
              if (iter2->weight>0.0) {
              if (P0>SHD.thetaP0) {
              if (*(iter2->input)>0 && P0>SHD.thetaP1) {
              // && *(iter2->P1)<SHD.thetaP0)
              iter2->weight += (1.0-iter2->weight/10.0) * SHD.eta_inhib * P0;
              if (iter2->weight>10.0) iter2->weight=10.0;
              }
              if (burst>0 && P0<SHD.thetaP1) {
              // && *(iter2->P1)>SHD.thetaP1)
              if (*(iter2->prePlastic)>0)
	      iter2->weight -= SHD.eta_disinhib * 
              *(iter2->prePlastic) * *(iter2->P1);
              else
	      iter2->weight -= SHD.eta_disinhib * *(iter2->P1);
              if (iter2->weight<0.0) iter2->weight=0.0;
              }
              }
              else {
              if (drandom(rng)<SHD.p_c) {
              iter2->weight = 0.001;
              }
              }
              }*/ //NOTE: this code was a plastic winnerless lateral connections attempt implemented by James K.
      }
  }
  
  // Update stim if using "internal" stimulus
  if (SHD.Bstim)
    {
      if (getSimulation().getIteration()%SHD.Dstim==0)
        if (++Cstim>=SHD.Nstim)
          Cstim=0;
      x += stim[Cstim];
    }
  
  // FN update
  double xprev=x;
  x += SHD.step1*(x-(x*x*x)/3-y-z*(x-SHD.nu)+0.35+R+drive);
  y += SHD.step2*(xprev+SHD.a-SHD.b*y);
  z += SHD.step3*(IN-z);

  if (SHD.corticostriatalPlasticity)
    {
      // SNc inputs - update dopamine modulation
      ShallowArray<SpikeInput>::iterator iter3, end3=dopamineInputs.end();
      for (iter3=dopamineInputs.begin(); iter3!=end3; ++iter3) {
        if (*(iter3->spike))
          iter3->weight = 1.0; // dopamine transient
        else if (iter3->weight > -1.0)
          iter3->weight -= 1.0/SHD.tauDA; // linear decrease dopamine in space
      }

      // Dopaminergic modulation at corto-striatal synapses by SNc neurons 
      double newBurst = (x>0) ? x : 0.0;
      double dW_STDP = newBurst-burst;
      if (dW_STDP != 0)
        dW_STDP = (dW_STDP>0) ? 1.0 : -1.0;
      double dW_GABA = (z>GABAt) ? -1.0 : 1.0;
      if (dopamineInputs.size()>0) {
        for (iter1=corticalInputs.begin(); iter1!=end1; ++iter1) {
          if (iter1->synapse) {
            if (*(iter1->spike)) {
              double dW_DA=iter1->modulator->weight; // amount of dopamine in extra-space
              if (D==1)
                dW_DA = (dW_DA>0) ? 1.0 : 0.0; // D1
              else { // D2
                if (dW_GABA>0) {
                  if (dW_STDP>0)
                    dW_DA = (dW_DA>0) ? 1.0 : 0.0;
                  else
                    dW_DA = (dW_DA<0) ? 1.0 : 0.0;
                }
                else {
                  if (dW_STDP>0)
                    dW_DA = (dW_DA<0) ? 1.0 : 0.0;
                  else
                    dW_DA = (dW_DA>0) ? 1.0 : 0.0;
                }
              }

              // update weight          
              iter1->weight += SHD.etaW * dW_GABA * dW_STDP * dW_DA;

              // synapse pruning and ...
              if (iter1->weight<=0) {
                iter1->weight=0;
                iter1->synapse=false;
              }
            }
          }
          else {
            // ... new synapses
            if (drandom(rng)<SHD.p_c) {
              iter1->synapse = true;
              iter1->weight = 0.001;
            }
          }
        }
      }
    }
}

void RabinovichWinnerlessUnit::copy(RNG& rng) 
{
  burstPrev = burst;
  burst = (x>0) ? x : 0.0;
  P1 = P0; // data independent copy of plasticity trace
}

void RabinovichWinnerlessUnit::outputWeights(std::ofstream& fsLN, std::ofstream& fsDR, std::ofstream& fsNS)
{
  //ShallowArray<PlasticInput>::iterator iter1, end1=lateralInputs.end(); // TODO: plasticity?
  ShallowArray<ModulatedSynapseInput>::iterator iter1, end1=corticalInputs.end();
  ShallowArray<StructuralInput>::iterator iter2, end2=lateralInputs.end();
  ShallowArray<SpikeInput>::iterator iter3, end3=dopamineInputs.end();

  for (iter1=corticalInputs.begin(); iter1!=end1; ++iter1)
    if (iter1->synapse) // only save if "functional"
      fsDR<<iter1->row<<" "<<iter1->col<<" "<<iter1->weight<<std::endl;  

  for (iter2=lateralInputs.begin(); iter2!=end2; ++iter2)
    if (iter2->synapse) // only save if "functional"
      fsLN<<iter2->row<<" "<<iter2->col<<" "<<iter2->weight<<std::endl;

  for (iter3=dopamineInputs.begin(); iter3!=end3; ++iter3)
    fsNS<<iter3->row<<" "<<iter3->col<<" "<<iter3->weight<<std::endl;
}

void RabinovichWinnerlessUnit::assymetric(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_RabinovichWinnerlessUnitInAttrPSet* CG_inAttrPset, CG_RabinovichWinnerlessUnitOutAttrPSet* CG_outAttrPset) 
{
  // N.B.: this function assumes both layers are in the same grid and therefore
  // of the same size.

  // If checking assymetry, check in and out.
  if (SHD.assymetric)
    {
      // Scoped variables
      RNG rng; // use a temporary local version only here: TODO update, not efficient on the GPU.
      unsigned connectionSeed=getSimulation().getRandomSeed();
      // Properties of the node
      unsigned n=getGridLayerData()->getNbrUnits();
      unsigned inIdx=CG_node->getNode()->getIndex();
      double frac = CG_inAttrPset->connectionFraction;
      unsigned outIdx=getNode()->getIndex();
      // (seed for the in->out connection)
      rng.reSeedShared(connectionSeed + (outIdx*n)+inIdx);
      double in = drandom(rng);    
      // If there should be a in->out connection ...
      if (in < frac) {
        // (seed for the out->in connection)
        rng.reSeedShared(connectionSeed +(inIdx*n)+outIdx);
        double out = drandom(rng);    
        // ... check if there would also be a out->in conection, and ... 
        if (out < frac)
          if (in < out) // ... if so, choose the in instead of the out so return true ...
            lateralInputs[lateralInputs.size()-1].synapse = true;
          else
            lateralInputs[lateralInputs.size()-1].synapse = false; // ... or false, and ...
        else
          lateralInputs[lateralInputs.size()-1].synapse = true; // ... if not, return true to allow the in->out connection.
      }
      else // ... or if not a in->out connection, return false.
        lateralInputs[lateralInputs.size()-1].synapse = false;
    }
  // Not checking assymetry so just work out if a connection  .
  else
    {
      RNG rng; // use a temporary local version only here: TODO update, not efficient on the GPU.
      // (seed for the in->out connection)
      rng.reSeedShared(getSimulation().getRandomSeed()
                       + (getNode()->getIndex()
                          * getGridLayerData()->getNbrUnits())
                       + CG_node->getNode()->getIndex());
      if (drandom(rng) < CG_inAttrPset->connectionFraction)        
        lateralInputs[lateralInputs.size()-1].synapse = true;
      else
        lateralInputs[lateralInputs.size()-1].synapse = false;
    }
}

void RabinovichWinnerlessUnit::checkForCorticalSynapse(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_RabinovichWinnerlessUnitInAttrPSet* CG_inAttrPset, CG_RabinovichWinnerlessUnitOutAttrPSet* CG_outAttrPset) 
{
  if (CG_inAttrPset->connection < CG_inAttrPset->connectionFraction)
    corticalInputs[corticalInputs.size()-1].synapse=true;
  else
    corticalInputs[corticalInputs.size()-1].weight=0.0;
  
  corticalInputs[corticalInputs.size()-1].row =  getGlobalIndex()+1; // +1 is for Matlab 
  corticalInputs[corticalInputs.size()-1].col = CG_node->getGlobalIndex()+1;
}

void RabinovichWinnerlessUnit::setLateralIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_RabinovichWinnerlessUnitInAttrPSet* CG_inAttrPset, CG_RabinovichWinnerlessUnitOutAttrPSet* CG_outAttrPset) 
{
  lateralInputs[lateralInputs.size()-1].row =  getGlobalIndex()+1; // +1 is for Matlab 
  lateralInputs[lateralInputs.size()-1].col = CG_node->getGlobalIndex()+1;
}

void RabinovichWinnerlessUnit::setModulatoryIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_RabinovichWinnerlessUnitInAttrPSet* CG_inAttrPset, CG_RabinovichWinnerlessUnitOutAttrPSet* CG_outAttrPset) 
{
  dopamineInputs[dopamineInputs.size()-1].row =  getGlobalIndex()+1; // +1 is for Matlab 
  dopamineInputs[dopamineInputs.size()-1].col = CG_node->getGlobalIndex()+1;
}

RabinovichWinnerlessUnit::~RabinovichWinnerlessUnit() 
{
}
