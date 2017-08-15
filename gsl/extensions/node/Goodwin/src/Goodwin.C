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
  // Set up any model specific instance variables
  if (SHD.op_Cannabinoids)
    k5_instance = SHD.k5;
}

void Goodwin::update(RNG& rng) 
{
  if (SHD.op_Cannabinoids)
    {      
      // Use an instance version of k5 which is updated with the amount of
      // unbound CB1R from the bouton
      X += ( (SHD.k1 / (SHD.K1 + pow(Z, SHD.n))) - (SHD.k2 * X) ) * SHD.deltaT;
      Y += ( (SHD.k3 * X) - (SHD.k4 * Y) ) * SHD.deltaT;
      Z += ( (k5_instance * Y) - (SHD.k6 * Z) ) * SHD.deltaT;
      if (in.size() > 0)
        k5_instance -= (( ((0.1-(*(in[0].input) * in[0].weight)) * // only consider first one, weight is structural plasticity
                           (k5_instance-SHD.k5max)) / SHD.k5tau) * SHD.deltaT);
    }
  else
    {
      // Typical equations for Goodwin model
      X += ( (SHD.k1 / (SHD.K1 + pow(Z, SHD.n))) - (SHD.k2 * X) ) * SHD.deltaT;
      Y += ( (SHD.k3 * X) - (SHD.k4 * Y) ) * SHD.deltaT;
      Z += ( (SHD.k5 * Y) - (SHD.k6 * Z) ) * SHD.deltaT;  
    }
}

void Goodwin::setInIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_GoodwinInAttrPSet* CG_inAttrPset, CG_GoodwinOutAttrPSet* CG_outAttrPset) 
{
}

Goodwin::~Goodwin() 
{
}

