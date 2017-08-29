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
#include "Goodwin.h"
#include "CG_Goodwin.h"
#include "rndm.h"

#define SHD getSharedMembers()

void Goodwin::initialize(RNG& rng)
{
  // Check if more than one input
  if (in1.size() != 1)
    assert("Goodwin: inputs should be one.");
  // Set up any model specific instance variables
  if (SHD.op_Cannabinoids)
    Cannabinoids_k1_instance = SHD.k1;
}

void Goodwin::update(RNG& rng) 
{
  if (SHD.op_Cannabinoids)
    {      
      // Use the unbound CB1R (i.e. Y-ECB) as the coupling between Y and Z and an instance version
      // of k1 which can be modified to change the CB1 mRNA max as observed in HD
      X += ( ( (Cannabinoids_k1_instance / (SHD.K1 + pow(Z, SHD.n))) - (SHD.k2 * X) ) / SHD.tau ) * SHD.deltaT;
      Y += ( ( (SHD.k3 * X) - (SHD.k4 * Y) ) / SHD.tau ) * SHD.deltaT;
      double Y_minus_ECB = Y;
      if (in1.size() > 0)
        Y_minus_ECB = Cannabinoids_Y_minus_ECB_sigmoid(Y - (*(in1[0].input) * in1[0].weight)); // only consider first one, weight is scaling to Goodwin model 'ECB'
      Z += ( ( (SHD.k5 * Y_minus_ECB) - (SHD.k6 * Z) ) / SHD.tau ) * SHD.deltaT;  
    }
  else
    {
      // Typical equations for Goodwin model
      X += ( ( (SHD.k1 / (SHD.K1 + pow(Z, SHD.n))) - (SHD.k2 * X) ) / SHD.tau ) * SHD.deltaT;
      Y += ( ( (SHD.k3 * X) - (SHD.k4 * Y) ) / SHD.tau ) * SHD.deltaT;
      Z += ( ( (SHD.k5 * Y) - (SHD.k6 * Z) ) / SHD.tau ) * SHD.deltaT;  
    }
}

void Goodwin::setInIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_GoodwinInAttrPSet* CG_inAttrPset, CG_GoodwinOutAttrPSet* CG_outAttrPset) 
{
  in1[in1.size()-1].row =  getIndex()+1; // +1 is for Matlab
  in1[in1.size()-1].col = CG_node->getIndex()+1;
}

Goodwin::~Goodwin() 
{
}

double Goodwin::Cannabinoids_Y_minus_ECB_sigmoid(double Y_minus_ECB)
{
  return 1.0 / ( 1.0 + exp(-SHD.Cannabinoids_sigmoid_C * (Y_minus_ECB - SHD.Cannabinoids_sigmoid_D)) );
}

