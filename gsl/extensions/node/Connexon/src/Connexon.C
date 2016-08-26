// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "Lens.h"
#include "Connexon.h"
#include "CG_Connexon.h"
#include "rndm.h"

void Connexon::produceInitialVoltage(RNG& rng) 
{
}

void Connexon::produceVoltage(RNG& rng) 
{
}

void Connexon::computeState(RNG& rng) 
{
#ifdef CONSIDER_MANYSPINE_EFFECT
  I=g*(*Vj-*Vi) / *countGapJunctionConnectedToCompartment_j;//TUAN TODO TO BE REVISED
#else
  I=g*(*Vj-*Vi);
#endif
}

void Connexon::setPointers(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_ConnexonInAttrPSet* CG_inAttrPset, CG_ConnexonOutAttrPSet* CG_outAttrPset) 
{
  index=CG_inAttrPset->idx;
  assert(getSharedMembers().voltageConnect);
  assert(index>=0 && index<getSharedMembers().voltageConnect->size());    
  Vi = &((*(getSharedMembers().voltageConnect))[index]);
#ifdef CONSIDER_MANYSPINE_EFFECT
  countGapJunctionConnectedToCompartment_i = 
    &((*(getSharedMembers().countGapJunctionConnect))[index]);
#endif
}

Connexon::~Connexon() 
{
}

