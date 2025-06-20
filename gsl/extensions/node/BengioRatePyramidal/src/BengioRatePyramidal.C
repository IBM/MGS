// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#include "Mgs.h"
#include "BengioRatePyramidal.h"
#include "CG_BengioRatePyramidal.h"
#include "rndm.h"

#define SHD getSharedMembers()

void BengioRatePyramidal::update_U(RNG& rng) 
{
  u += SHD.dT * (-SHD.g_lk * u + SHD.g_B *  (v_B - u) + 
		 SHD.g_A * (v_A - u) + SHD.i_toggle * i + SHD.sigma * gaussian(rng) );
  phi_u = 1.0/(1.0+exp(-u));

  i = 0;
  ShallowArray<double*>::iterator iter, end=pyramidalTeachingInputs.end();
  for (iter=pyramidalTeachingInputs.begin(); iter!=end; ++iter) {
    double g_Exc = SHD.g_som * (**iter - SHD.E_inh) / (SHD.E_exc - SHD.E_inh);
    double g_Inh = SHD.g_som * (**iter - SHD.E_exc) / (SHD.E_exc - SHD.E_inh);
    i += g_Exc * (SHD.E_exc - u) + g_Inh * (SHD.E_inh - u);
  }
}

void BengioRatePyramidal::update_Vs(RNG& rng) // Vs is plural (V_A and V_B)
{
  ShallowArray<Input>::iterator iter, end=pyramidalForwardInputs.end();
  double d_rate = phi_u - (1.0 / (1.0+exp(-SHD.predictionFactor*v_B)));
  v_B = 0;
  for (iter=pyramidalForwardInputs.begin(); iter!=end; ++iter) {
    v_B += iter->weight * ( *(iter->input) );
    iter->weight += SHD.eta_PP * d_rate * ( *(iter->input) );
  }
  
  end=pyramidalBackwardInputs.end();
  double v_A_prev = v_A, v_TD = 0;
  v_A = 0;
  for (iter=pyramidalBackwardInputs.begin(); iter!=end; ++iter) {
    v_TD = iter->weight * ( *(iter->input) );
    v_A += v_TD;
    iter->weight += SHD.eta_PP * (phi_u - (1.0 / (1.0+exp(-v_TD) ) ) ) * ( *(iter->input) );
    // Note: logistic function phi is applied twice to the backward input for weight update
    //       (which is dubious, see paper and consider revision)
  }
  end=interneuronInputs.end();
  for (iter=interneuronInputs.begin(); iter!=end; ++iter) {
    v_A += iter->weight * ( *(iter->input) );
    iter->weight += SHD.eta_PI * (SHD.v_rest - v_A_prev) * ( *(iter->input) );
  }
}

void BengioRatePyramidal::setLateralIndices(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_BengioRatePyramidalInAttrPSet* CG_inAttrPset, CG_BengioRatePyramidalOutAttrPSet* CG_outAttrPset) 
{
}

BengioRatePyramidal::~BengioRatePyramidal() 
{
}

