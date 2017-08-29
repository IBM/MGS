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
#include "MihalasNieburSynapseIAFUnit.h"
#include "CG_MihalasNieburSynapseIAFUnit.h"
#include "rndm.h"

#define SHD getSharedMembers()

void MihalasNieburSynapseIAFUnit::initialize(RNG& rng)
{
  //  std::cout << AMPAcurrentInputs.size() << std::endl;
  spike=false;
  V=SHD.V_r;
  Theta=SHD.Theta_inf;
  V_spike=V;
  int nI=SHD.k.size();
  I.increaseSizeTo(nI);
  I_p.increaseSizeTo(2);
  I_p[0].increaseSizeTo(nI);
  I_p[1].increaseSizeTo(nI);
  dI.increaseSizeTo(nI);
  V_p.increaseSizeTo(2);
  Theta_p.increaseSizeTo(2);
  for (int n=0; n<nI; ++n) I[n]=I_p[0][n]=I_p[1][n]=dI[n]=0;
  for (int n=0; n<2; ++n) V_p[n]=Theta_p[n]=0;
}

void MihalasNieburSynapseIAFUnit::update(RNG& rng)
{
  // Synapses
  double s_total = 0.;
  ShallowArray<SynapticCurrentIAFInput>::iterator iter, end=AMPAcurrentInputs.end();
  for (iter=AMPAcurrentInputs.begin(); iter!=end; ++iter)
    s_total += *(iter->AMPAcurrent) * iter->weight; // weight is structural plasticity

  // Neuron
  double I_e = s_total; // total input
  int nI=I.size();
  double I_sum = 0.;
  int ip=0;
  for (int n=0; n<nI; ++n) {
    dI[n] = SHD.k[n]*I[n]*SHD.deltaT;
    I_p[ip][n] = I[n] - dI[n];
    I_sum = I_sum + I[n];
  }
  double dV = (1/SHD.C*(I_e+I_sum)-SHD.G*(V-SHD.E_L))*SHD.deltaT;
  V_p[ip] = V + dV;
  double dTheta = (SHD.a*(V-SHD.E_L)-SHD.b*(Theta-SHD.Theta_inf))*SHD.deltaT;
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
    Theta_p[ip] = Theta + 0.5*(dTheta + (SHD.a*(V_p[ip_prime]-SHD.E_L)-SHD.b*(Theta_p[ip_prime]-SHD.Theta_inf))*SHD.deltaT);
  }
  for (int n=0; n<nI; ++n)
    I[n]=I_p[ip][n];
  V=V_p[ip];
  Theta=Theta_p[ip];
}

void MihalasNieburSynapseIAFUnit::threshold(RNG& rng)
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

void MihalasNieburSynapseIAFUnit::setAMPAIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_MihalasNieburSynapseIAFUnitInAttrPSet* CG_inAttrPset, CG_MihalasNieburSynapseIAFUnitOutAttrPSet* CG_outAttrPset)
{
  AMPAcurrentInputs[AMPAcurrentInputs.size()-1].row =  getIndex()+1; // +1 is for Matlab
  AMPAcurrentInputs[AMPAcurrentInputs.size()-1].col = CG_node->getIndex()+1;
}

MihalasNieburSynapseIAFUnit::~MihalasNieburSynapseIAFUnit()
{
}
