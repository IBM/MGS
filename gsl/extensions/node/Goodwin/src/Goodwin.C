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

void Goodwin::update(RNG& rng) 
{
  // Typical equations for Goodwin model
  X += ( (SHD.k1 / (SHD.K1 + pow(Z, SHD.n))) - (SHD.k2 * X) ) * SHD.deltaT;
  Y += ( (SHD.k3 * X) - (SHD.k4 * Y) ) * SHD.deltaT;
  Z += ( (SHD.k5 * Y) - (SHD.k6 * Z) ) * SHD.deltaT;  
}

void Goodwin::setInIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_GoodwinInAttrPSet* CG_inAttrPset, CG_GoodwinOutAttrPSet* CG_outAttrPset) 
{
}

Goodwin::~Goodwin() 
{
}

