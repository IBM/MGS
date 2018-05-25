// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2018
//
// (C) Copyright IBM Corp. 2005-2018  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "Lens.h"
#include "FSIIAFUnit.h"
#include "CG_FSIIAFUnit.h"
#include "GridLayerData.h"
#include "rndm.h"
#include <fstream>
#include <sstream>

#define SHD getSharedMembers()
#define ITER getSimulation().getIteration()
#define TIME ITER*getSharedMembers().deltaT
#define RANK getSimulation().getRank()

void FSIIAFUnit::initialize(RNG& rng) 
{
  spike=false;
  //  Theta=SHD.Theta_inf;
  Theta=Theta_inf;
  V_spike=V;
  int nI=SHD.k.size();
  I.increaseSizeTo(nI);
  I_p.increaseSizeTo(2);
  I_p[0].increaseSizeTo(nI);
  I_p[1].increaseSizeTo(nI);
  dI.increaseSizeTo(nI);
  V_p.increaseSizeTo(2);
  Theta_p.increaseSizeTo(2);
  for (int n=0; n<nI; ++n)
    I[n]=I_p[0][n]=I_p[1][n]=dI[n]=0;
  for (int n=0; n<2; ++n)
    V_p[n]=Theta_p[n]=0;
  // If this node receives multiple cortical inputs, they should
  // be weighted accordingly so they contribute the same as another
  // node which only receives one.
  if (ctxInputs.size() > 1)
    ctxInputWeight = 1.0 / (double) ctxInputs.size();
  else
    ctxInputWeight = 1.0;
}

void FSIIAFUnit::updateInput(RNG& rng) 
{
  // Cortex input
  double driver = 0.0;
  ShallowArray<Input>::iterator iterDriver, endDriver=ctxInputs.end();
  for (iterDriver=ctxInputs.begin(); iterDriver!=endDriver; ++iterDriver)
    driver += (*(iterDriver->input) * iterDriver->weight) * ctxInputWeight;

  // Synapses
  double s_total = 0.0;
  ShallowArray<PSPInput>::iterator iterIPSP, endPSP=lateralInputs.end();
  for (iterIPSP=lateralInputs.begin(); iterIPSP!=endPSP; ++iterIPSP)
    {      
      iterIPSP->s_r = iterIPSP->s_r +
        (((-iterIPSP->s_r + (*(iterIPSP->spike) ? 1.0 : 0.0)) / SHD.s_tauR)
         * SHD.deltaT);
      iterIPSP->s_f = iterIPSP->s_f +
        (((-iterIPSP->s_f + iterIPSP->s_r) / SHD.s_tauF)
         * SHD.deltaT);
      s_total += iterIPSP->s_f * iterIPSP->weight;
    }
  
  // Gap junctions
  ShallowArray<GJInput>::iterator iterGJ, endGJ=gjInputs.end();
  double etonic=0.;
  for (iterGJ=gjInputs.begin(); iterGJ!=endGJ; ++iterGJ) {
    etonic += (*(iterGJ->voltage)-V)*iterGJ->conductance;
  }

  I_e = driver + s_total + etonic; // total input
  synapses_total = driver + s_total; // total synaptic input - excitatory drive and lateral inhibition
  if (SHD.op_saveExtra)
    {
      drive_total = driver; // total excitatory input - excitatory drive
      IPSC_total = s_total; // total IPSC input - lateral inhibition
      GJ_total = etonic; // total GJ input - lateral GJs
    }
}

void FSIIAFUnit::updateV(RNG& rng)
{
  // Neuron
  int nI=I.size();
  double I_sum = 0.0;
  int ip=0;
  for (int n=0; n<nI; ++n) {
    dI[n] = SHD.k[n]*I[n]*SHD.deltaT;
    I_p[ip][n] = I[n] - dI[n];
    I_sum = I_sum + I[n];
  }
  double dV = (1/SHD.C*(I_e+I_sum)-SHD.G*(V-SHD.E_L))*SHD.deltaT;
  V_p[ip] = V + dV; 
  double dTheta = (SHD.a*(V-SHD.E_L)-SHD.b*(Theta-Theta_inf))*SHD.deltaT;
  Theta_p[ip] = Theta + dTheta;
  /* Fixed Point Iteration */
  for (int p=0; p<SHD.np; ++p) {
    ip=((p+1)%2);
    int ip_prime=(p%2);
    double I_psum=0;
    for (int n=0; n<nI; ++n) {
      I_p[ip][n] = I[n] + 0.5*(dI[n] - (SHD.k[n])*I_p[ip_prime][n]*(SHD.deltaT));
      I_psum = I_psum + I_p[ip_prime][n];
    }
    V_p[ip] = V + 0.5*(dV + (1/SHD.C*(I_e+I_sum-SHD.G*(V_p[ip_prime]-SHD.E_L)))*SHD.deltaT);
    Theta_p[ip] = Theta + 0.5*(dTheta + (SHD.a*(V_p[ip_prime]-SHD.E_L)-SHD.b*(Theta_p[ip_prime]-Theta_inf))*SHD.deltaT);
  }
  for (int n=0; n<nI; ++n)
    I[n]=I_p[ip][n];
  V=V_p[ip];
  Theta=Theta_p[ip];
}

void FSIIAFUnit::threshold(RNG& rng) 
{
  spike=(V>Theta);
  if (spike)
    {
      int nI=I.size();
      for (int n=0; n<nI; ++n)
        I[n] = SHD.R[n]*I[n]+SHD.A[n];
      V = SHD.V_r;
      Theta = (Theta>SHD.Theta_r) ? Theta : SHD.Theta_r;
      V_spike=SHD.V_max;
      spike_cnt = 1;
    }
  else if ((spike_cnt < (int) (SHD.spike_cntMax / SHD.deltaT)) 
           && (spike_cnt > 0))
    {
      V_spike=SHD.V_max;
      spike_cnt++;
    }
  else if (spike_cnt >= (int) (SHD.spike_cntMax / SHD.deltaT))
    {
      spike_cnt = 0;
      V_spike=V;
    }
  else
    V_spike=V;
}

void FSIIAFUnit::outputPSPs(std::ofstream& fs)
{
  ShallowArray<PSPInput>::iterator iter, end=lateralInputs.end();
  float temp = 0.;
  for (iter=lateralInputs.begin(); iter!=end; ++iter)
    {
      temp = (float) iter->s_f;
      fs.write(reinterpret_cast<char *>(&temp), sizeof(temp));
    }
}

void FSIIAFUnit::outputWeights(std::ofstream& fs)
{
  ShallowArray<PSPInput>::iterator iter, end=lateralInputs.end();
  int temp = lateralInputs.size();
  fs.write(reinterpret_cast<char *>(&temp), sizeof(temp));  
  float temp2 = 0.;
  for (iter=lateralInputs.begin(); iter!=end; ++iter)
    {
      temp2 = (float) iter->weight;
      fs.write(reinterpret_cast<char *>(&temp2), sizeof(temp2));
    }
}

void FSIIAFUnit::outputGJs(std::ofstream& fs)
{
  ShallowArray<GJInput>::iterator iter, end=gjInputs.end();
  int temp = gjInputs.size();  
  fs.write(reinterpret_cast<char *>(&temp), sizeof(temp));  
  float temp2 = 0.;
  for (iter=gjInputs.begin(); iter!=end; ++iter)
    {
      temp2 = (float) iter->conductance;
      fs.write(reinterpret_cast<char *>(&temp2), sizeof(temp2));
      /*
      temp2 = (float) iter->row;
      fs.write(reinterpret_cast<char *>(&temp2), sizeof(temp2));
      temp2 = (float) iter->col;
      fs.write(reinterpret_cast<char *>(&temp2), sizeof(temp2));
      */
    }
}

bool FSIIAFUnit::bidirectional1(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_FSIIAFUnitInAttrPSet* CG_inAttrPset, CG_FSIIAFUnitOutAttrPSet* CG_outAttrPset) 
{
  
  // N.B.: this function assumes both layers are in the same grid and therefore
  // of the same size.

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
  if (in <= frac)
    return true;
  else
    return false;
}

bool FSIIAFUnit::bidirectional2(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_FSIIAFUnitInAttrPSet* CG_inAttrPset, CG_FSIIAFUnitOutAttrPSet* CG_outAttrPset) 
{
  
  // N.B.: this function assumes both layers are in the same grid and therefore
  // of the same size.
  
  // Scoped variables
  RNG rng; // use a temporary local version only here: TODO update, not efficient on the GPU.
  unsigned connectionSeed=getSimulation().getRandomSeed();
  // Properties of the node
  unsigned n=getGridLayerData()->getNbrUnits();
  unsigned inIdx=CG_node->getNode()->getIndex();
  double frac = CG_inAttrPset->connectionFraction;
  unsigned outIdx=getNode()->getIndex();
  
  // (seed for the out->in connection)
  rng.reSeedShared(connectionSeed + (inIdx*n)+outIdx);
  double in = drandom(rng);
  
  // If there should be an out->in connection (because there was an in->out) ...
  if (in <= frac)
    {
      // If already bidirectional, don't connect
      rng.reSeedShared(connectionSeed + (outIdx*n)+inIdx);
      double in = drandom(rng);
      if (in <= frac)
        return false;
      else
        return true;
    }
  else
    return false;
}
 
void FSIIAFUnit::setIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_FSIIAFUnitInAttrPSet* CG_inAttrPset, CG_FSIIAFUnitOutAttrPSet* CG_outAttrPset) 
{
  if (CG_inAttrPset->identifier=="driver") {
    ctxInputs[ctxInputs.size()-1].row =  getGlobalIndex()+1; // +1 is for Matlab 
    ctxInputs[ctxInputs.size()-1].col = CG_node->getGlobalIndex()+1;   
  } else if (CG_inAttrPset->identifier=="ipsp") {
    lateralInputs[lateralInputs.size()-1].row =  getGlobalIndex()+1; // +1 is for Matlab 
    lateralInputs[lateralInputs.size()-1].col = CG_node->getGlobalIndex()+1;   
  } else if ((CG_inAttrPset->identifier=="gj1") || (CG_inAttrPset->identifier=="gj2")) {
    gjInputs[gjInputs.size()-1].row =  getGlobalIndex()+1; // +1 is for Matlab 
    gjInputs[gjInputs.size()-1].col = CG_node->getGlobalIndex()+1;   
  }
}

FSIIAFUnit::~FSIIAFUnit() 
{
}

