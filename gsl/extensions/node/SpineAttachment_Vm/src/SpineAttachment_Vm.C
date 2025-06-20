// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Mgs.h"
#include "SpineAttachment_Vm.h"
#include "CG_SpineAttachment_Vm.h"
#include "rndm.h"
#include <cmath>
#include <cfloat>
#include "NTSMacros.h"

SpineAttachment_Vm::SpineAttachment_Vm() { _gotAssigned = false; }

void SpineAttachment_Vm::produceInitialState(RNG& rng) {}

void SpineAttachment_Vm::computeInitialState(RNG& rng)
{
  assert(Vi);
  assert(Vj);
  // NOTE: g (nS) which is infered from R (GigaOhm)
  //   R = rho. l / A
  //   rho (GigaOhm.cm) = specific resisitivity
  //   l = length of diffusion
  //   A = cross-sectional area
  //   r1 = radius of spine neck
  //   r2 = radius of dendritic shaft
  //  with complex geometry, calculating the resistance be much more complicated
  //  https://en.wikipedia.org/wiki/Electrical_resistivity_and_conductivity#Resistance_versus_resistivity_in_complicated_geometries
  //  SOLUTION: l = distance from 2 center points in neck + compartment
  //     l = 1/2 neck-length + r2
  //            A = pi * ((r1+r2)/2)^2
  //            rho = Ra ~ 100 GOhm.um
  //  g = 1/R = A / (rho * l)
  dyn_var_t A = std::abs(Ai - *Aj); //[um^2]
  dyn_var_t distance;
  CustomString typeDenShaft("den-shaft");
  CustomString typeSpineNeck("spine-neck");
  if (typeCpt == typeDenShaft)
  {
   distance = (*leni + *lenj / 2.0); //[um]
  }else{//connect to spine-neck
   distance = (*leni / 2.0  + *lenj); //[um]
  }
#ifdef CONSIDER_MANYSPINE_EFFECT_OPTION2
  {
  g = A / (Raxial * distance)/ (dimension->surface_area);            // [nS/um^2]
#ifdef CONSIDER_MANYSPINE_EFFECT_OPTION2_revised
  //g = g / *countSpineConnectedToCompartment_j; 
  g = g / *countSpineConnectedToCompartment_i;  // we change the reduction from the side that has man contacts, e.g. den-shaft only
#endif
  }
#else
  g = A / (Raxial * distance);            // [nS]
#endif
}

void SpineAttachment_Vm::produceState(RNG& rng) {}

void SpineAttachment_Vm::computeState(RNG& rng)
{
	//i = index of compartment this connexon is connecting to
	//j = index of compartment from the other side
  dyn_var_t V = *Vj - *Vi;
#ifdef CONSIDER_MANYSPINE_EFFECT_OPTION1
  //I = g * V / *countSpineConnectedToCompartment_j;
  I = g * V / *countSpineConnectedToCompartment_i; //[pA]
#else
#ifdef CONSIDER_MANYSPINE_EFFECT_OPTION2
  //no need to update I(Vm), as it produces g, and Vj
#else
  I = g * V;  // g = should be a function of spineneck size
#endif
#endif
}

void SpineAttachment_Vm::setVoltagePointers(
    const CustomString& CG_direction, const CustomString& CG_component,
    NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
    Constant* CG_constant, CG_SpineAttachment_VmInAttrPSet* CG_inAttrPset,
    CG_SpineAttachment_VmOutAttrPSet* CG_outAttrPset)
{
  if (_gotAssigned)
    assert(index == CG_inAttrPset->idx);
  else
  {
    index = CG_inAttrPset->idx;
    _gotAssigned = true;
  }
  assert(getSharedMembers().voltageConnect);
  assert(index >= 0 && index < getSharedMembers().voltageConnect->size());
  Vi = &((*(getSharedMembers().voltageConnect))[index]);
#ifdef CONSIDER_MANYSPINE_EFFECT_OPTION2
  dimension = ((*(getSharedMembers().dimensionsConnect))[index]);
#endif
//#ifdef CONSIDER_MANYSPINE_EFFECT_OPTION1
#if defined(CONSIDER_MANYSPINE_EFFECT_OPTION1) || defined(CONSIDER_MANYSPINE_EFFECT_OPTION2_revised)
  unsigned size = getSharedMembers().voltageConnect->size() ;  //# of compartments
  if ((*(getSharedMembers().countSpineConnect)).size() != size) 
  {
    (*(getSharedMembers().countSpineConnect)).increaseSizeTo(size);
    for (int i = 0; i < size; i++)
      (*(getSharedMembers().countSpineConnect))[i] = 0;
  }
  countSpineConnectedToCompartment_i = &((*(getSharedMembers().countSpineConnect))[index]);
#endif
}

void SpineAttachment_Vm::set_A_and_len(
    const CustomString& CG_direction, const CustomString& CG_component,
    NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
    Constant* CG_constant, CG_SpineAttachment_VmInAttrPSet* CG_inAttrPset,
    CG_SpineAttachment_VmOutAttrPSet* CG_outAttrPset)
{
  if (_gotAssigned)
    assert(index == CG_inAttrPset->idx);
  else
  {
    index = CG_inAttrPset->idx;
    _gotAssigned = true;
  }
  assert(getSharedMembers().dimensionsConnect);
  CustomString cptType = CG_inAttrPset->typeCpt;
  CustomString typeDenShaft("den-shaft");
  CustomString typeSpineNeck("spine-neck");
  DimensionStruct* dimension = (*(getSharedMembers().dimensionsConnect))[index];
  if (cptType == typeDenShaft)
  {
    // len2 = radius of the shaft
    // A2   = zero (from shaft-side)
    Ai = 0.0;
    leni = &(dimension->r);
    typeCpt = typeDenShaft;
  }
  else if (cptType == typeSpineNeck)
  {
    // len1 = 1/2 spineneck (from spineneck-side)
    // A1   = cross-sectional surface area of spineneck
    _ri = &(dimension->r);
    Ai = M_PI * (*_ri) * (*_ri);
    leni = &(dimension->length);
    typeCpt = typeDenShaft;
  }
  else{//do not accept other names
    assert(0);
  }
}

SpineAttachment_Vm::~SpineAttachment_Vm() {}
