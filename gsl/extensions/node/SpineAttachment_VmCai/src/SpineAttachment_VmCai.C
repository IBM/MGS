// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "Lens.h"
#include "SpineAttachment_VmCai.h"
#include "CG_SpineAttachment_VmCai.h"
#include "rndm.h"
#include <cmath>
#include <cfloat>
#include "NTSMacros.h"

SegmentDescriptor SpineAttachment_VmCai::_segmentDescriptor;

SpineAttachment_VmCai::SpineAttachment_VmCai() 
{
	_gotAssigned = false;
}

void SpineAttachment_VmCai::produceInitialState(RNG& rng)
{

}
void SpineAttachment_VmCai::computeInitialState(RNG& rng)
{
#ifdef DEBUG_COMPARTMENT
  volatile unsigned nidx = _segmentDescriptor.getNeuronIndex(branchData->key);
  volatile unsigned bidx = _segmentDescriptor.getBranchIndex(branchData->key);
  volatile unsigned iteration = getSimulation().getIteration();
  volatile unsigned nidxOther = _segmentDescriptor.getNeuronIndex(branchDataOther->key);
  volatile unsigned bidxOther = _segmentDescriptor.getBranchIndex(branchDataOther->key);
#endif
  assert(Vi);
  assert(Vj);
  assert(Cai);
  assert(Caj);
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
  dyn_var_t distance;
  String typeDenShaft("den-shaft");
  String typeSpineNeck("spine-neck");
  if (typeCpt == typeDenShaft)
  {
   distance = (*leni + *lenj / 2.0); //[um]
  }else{//connect to spine-neck
   distance = (*leni / 2.0  + *lenj); //[um]
  }
#ifdef CONSIDER_MANYSPINE_EFFECT_OPTION2
  g = A / (Raxial * distance)/ (dimension->surface_area);            // [nS/um^2]
#else
  g = A / (Raxial * distance);            // [nS]
#endif
  //TUAN TODO: we haven't consider the effect of smaller cross-sectional area for cyto yet
#ifdef CONSIDER_MANYSPINE_EFFECT_OPTION2_CACYTO
  invTimeCacyto = A * DCa * (dimension->surface_area * FRACTION_SURFACEAREA_CYTO ) / 
    ((dimension->volume * FRACTIONVOLUME_CYTO) * distance); //[1/ms]
#else
  Caconc2current = A * DCa * zCa * zF / (1000000.0*distance); //[pA/uM]
#endif
}

void SpineAttachment_VmCai::produceState(RNG& rng) {}

void SpineAttachment_VmCai::computeState(RNG& rng)
{
#ifdef DEBUG_COMPARTMENT
  volatile unsigned nidx = _segmentDescriptor.getNeuronIndex(branchData->key);
  volatile unsigned bidx = _segmentDescriptor.getBranchIndex(branchData->key);
  volatile unsigned iteration = getSimulation().getIteration();
  volatile unsigned nidxOther = _segmentDescriptor.getNeuronIndex(branchDataOther->key);
  volatile unsigned bidxOther = _segmentDescriptor.getBranchIndex(branchDataOther->key);
#endif
	//i = index of compartment this connexon is connecting to
	//j = index of compartment from the other side
  dyn_var_t V = *Vj - *Vi;
#ifdef CONSIDER_MANYSPINE_EFFECT_OPTION1
  I = g * V / *countSpineConnectedToCompartment_j; //[pA]
  I_Ca = Caconc2current * (*Caj - *Cai)/ *countSpineConnectedToCompartment_j; //[pA]
#else
#ifdef CONSIDER_MANYSPINE_EFFECT_OPTION2_CACYTO
  //no need to update I(Vm), as it produces g, and Vj
  //no need to update Cacyto, as it produces invTimeCacyto, and Cacytoj
#else
  I = g * V;
  I_Ca = Caconc2current * (*Caj - *Cai);
#endif
#endif
}

void SpineAttachment_VmCai::setVoltagePointers(
    const String& CG_direction, const String& CG_component,
    NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
    Constant* CG_constant, CG_SpineAttachment_VmCaiInAttrPSet* CG_inAttrPset,
    CG_SpineAttachment_VmCaiOutAttrPSet* CG_outAttrPset)
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
  dimension = ((*(getSharedMembers().dimensionsConnect))[index]);
#ifdef CONSIDER_MANYSPINE_EFFECT_OPTION1
  countSpineConnectedToCompartment_i = &((*(getSharedMembers().countSpineConnect))[index]);
#endif
}

void SpineAttachment_VmCai::setCaPointers(
    const String& CG_direction, const String& CG_component,
    NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
    Constant* CG_constant, CG_SpineAttachment_VmCaiInAttrPSet* CG_inAttrPset,
    CG_SpineAttachment_VmCaiOutAttrPSet* CG_outAttrPset)
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


void SpineAttachment_VmCai::set_A_and_len(
    const String& CG_direction, const String& CG_component,
    NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
    Constant* CG_constant, CG_SpineAttachment_VmCaiInAttrPSet* CG_inAttrPset,
    CG_SpineAttachment_VmCaiOutAttrPSet* CG_outAttrPset)
{
#ifdef DEBUG_COMPARTMENT
  volatile unsigned nidx = _segmentDescriptor.getNeuronIndex(branchData->key);
  volatile unsigned bidx = _segmentDescriptor.getBranchIndex(branchData->key);
  volatile unsigned iteration = getSimulation().getIteration();
#endif
  if (_gotAssigned)
	  assert(index == CG_inAttrPset->idx);
  else
  {
	  index = CG_inAttrPset->idx;
	  _gotAssigned = true;
  }
  assert(getSharedMembers().dimensionsConnect);
  String cptType = CG_inAttrPset->typeCpt;
  String typeDenShaft("den-shaft");
  String typeSpineNeck("spine-neck");
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

SpineAttachment_VmCai::~SpineAttachment_VmCai() {}
