// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2015-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "Lens.h"
#include "SpineAttachment_VmCaiCaER.h"
#include "CG_SpineAttachment_VmCaiCaER.h"
#include "rndm.h"
#include <cmath>
#include <cfloat>
#include "NTSMacros.h"


SpineAttachment_VmCaiCaER::SpineAttachment_VmCaiCaER() 
{
	_gotAssigned = false;
}

void SpineAttachment_VmCaiCaER::produceInitialState(RNG& rng)
{
  assert(Vi);
  assert(Vj);
  assert(Cai);
  assert(Caj);
  assert(CaERi);
  assert(CaERj);
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
  assert(Raxial > MIN_RESISTANCE_VALUE);
  dyn_var_t A = std::abs(Ai - *Aj); //[um^2]
  dyn_var_t len = (*leni + *lenj) / 2.0; //[um]
  g = A / (Raxial * len);            // [nS]
  Caconc2current = A * DCa * zCa * zF / (1000000.0*len);
  CaERconc2current = A * DCaER * zCa * zF / (1000000.0*len);
  //TUAN TODO: HERE we should take into account the cross-sectional difference
  //A for Vm
  //A for Cacyto
  //A for CaER
  //
  
}

void SpineAttachment_VmCaiCaER::produceState(RNG& rng) {}

void SpineAttachment_VmCaiCaER::computeState(RNG& rng)
{
	//i = index of compartment this connexon is connecting to
	//j = index of compartment from the other side
  float V = *Vj - *Vi;
  I = g * V;
  I_Ca = Caconc2current * (*Caj - *Cai);
  I_CaER = CaERconc2current * (*CaERj - *CaERi);
  //std::cout << *Cai << "  " << *Caj << "; ER " << *CaERi << " " << *CaERj << "\n";
}

void SpineAttachment_VmCaiCaER::setVoltagePointers(
    const String& CG_direction, const String& CG_component,
    NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
    Constant* CG_constant,
    CG_SpineAttachment_VmCaiCaERInAttrPSet* CG_inAttrPset,
    CG_SpineAttachment_VmCaiCaEROutAttrPSet* CG_outAttrPset)
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
}

void SpineAttachment_VmCaiCaER::setCaPointers(
    const String& CG_direction, const String& CG_component,
    NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
    Constant* CG_constant,
    CG_SpineAttachment_VmCaiCaERInAttrPSet* CG_inAttrPset,
    CG_SpineAttachment_VmCaiCaEROutAttrPSet* CG_outAttrPset)
{
  if (_gotAssigned)
	  assert(index == CG_inAttrPset->idx);
  else
  {
	  index = CG_inAttrPset->idx;
	  _gotAssigned = true;
  }
  assert(getSharedMembers().CaConcentrationConnect);
  Cai = &((*(getSharedMembers().CaConcentrationConnect))[index]);
}

void SpineAttachment_VmCaiCaER::setCaERPointers(
    const String& CG_direction, const String& CG_component,
    NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
    Constant* CG_constant,
    CG_SpineAttachment_VmCaiCaERInAttrPSet* CG_inAttrPset,
    CG_SpineAttachment_VmCaiCaEROutAttrPSet* CG_outAttrPset)
{
  if (_gotAssigned)
	  assert(index == CG_inAttrPset->idx);
  else
  {
	  index = CG_inAttrPset->idx;
	  _gotAssigned = true;
  }
  assert(getSharedMembers().CaERConcentrationConnect);
  CaERi = &((*(getSharedMembers().CaERConcentrationConnect))[index]);
}

void SpineAttachment_VmCaiCaER::set_A_and_len(
    const String& CG_direction, const String& CG_component,
    NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
    Constant* CG_constant,
    CG_SpineAttachment_VmCaiCaERInAttrPSet* CG_inAttrPset,
    CG_SpineAttachment_VmCaiCaEROutAttrPSet* CG_outAttrPset)
{
  assert(getSharedMembers().dimensionsConnect);
  String cptType (CG_inAttrPset->typeCpt);
  String typeDenShaft("den-shaft");
  String typeSpineNeck("spine-neck");
  DimensionStruct* dimension = (*(getSharedMembers().dimensionsConnect))[0];
  if (cptType == typeDenShaft)
  {
    // len2 = radius of the shaft
    // A2   = zero (from shaft-side)
    Ai = 0.0;
    leni = &(dimension->r);
  }
  else if (cptType == typeSpineNeck)
  {
    // len1 = 1/2 spineneck (from spineneck-side)
    // A1   = cross-sectional surface area of spineneck
    _ri = &(dimension->r);
    Ai = M_PI * (*_ri) * (*_ri);
    leni = &(dimension->length);
  }
}

SpineAttachment_VmCaiCaER::~SpineAttachment_VmCaiCaER() {}
