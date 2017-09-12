// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
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
#ifdef CONSIDER_MANYSPINE_EFFECT_OPTION2
  g = g / (dimension->surface_area);            // [nS/um^2]
#ifdef CONSIDER_MANYSPINE_EFFECT_OPTION2_revised
  //g = g / *countGapJunctionConnectedToCompartment_j; 
  g = g / *countGapJunctionConnectedToCompartment_i; 
#endif
#else
  //g = A / (Raxial * distance);            // [nS]  -- provided by the user via parameter
#endif
}

void Connexon::produceVoltage(RNG& rng) 
{
}

void Connexon::computeState(RNG& rng) 
{
#ifdef CONSIDER_MANYSPINE_EFFECT_OPTION1
  I=g*(*Vj-*Vi) / *countGapJunctionConnectedToCompartment_j;//TUAN TODO TO BE REVISED
#else
#ifdef CONSIDER_MANYSPINE_EFFECT_OPTION2
  //no need to update I(Vm), as it produces g, and Vj
#else
  I=g*(*Vj-*Vi);
#endif
#endif
}

void Connexon::setPointers(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_ConnexonInAttrPSet* CG_inAttrPset, CG_ConnexonOutAttrPSet* CG_outAttrPset) 
{
  if (CG_inAttrPset->idx>=0) {
    index=CG_inAttrPset->idx;
  }
  else if (CG_inAttrPset->idx==-1) {
    index=int(float(branchData->size)*CG_inAttrPset->branchProp);
  }
  assert(getSharedMembers().voltageConnect);
  assert(index>=0 && index<getSharedMembers().voltageConnect->size());
  Vi = &((*(getSharedMembers().voltageConnect))[index]);
//#ifdef CONSIDER_MANYSPINE_EFFECT_OPTION1
#if defined(CONSIDER_MANYSPINE_EFFECT_OPTION1) || defined(CONSIDER_MANYSPINE_EFFECT_OPTION2_revised)
  countGapJunctionConnectedToCompartment_i = 
    &((*(getSharedMembers().countGapJunctionConnect))[index]);
#endif
#ifdef CONSIDER_MANYSPINE_EFFECT_OPTION2
  dimension = ((*(getSharedMembers().dimensionsConnect))[index]);
#endif
}

Connexon::~Connexon() 
{
}

