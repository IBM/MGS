#include "Lens.h"
#include "CaERConcentrationJunctionPoint.h"
#include "CG_CaERConcentrationJunctionPoint.h"
#include "rndm.h"

void CaERConcentrationJunctionPoint::produceInitialState(RNG& rng) 
{
}

void CaERConcentrationJunctionPoint::produceCaConcentration(RNG& rng) 
{
}

void CaERConcentrationJunctionPoint::setPointers(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CaERConcentrationJunctionPointInAttrPSet* CG_inAttrPset, CG_CaERConcentrationJunctionPointOutAttrPSet* CG_outAttrPset) 
{
  assert(getSharedMembers().CaConcentrationConnect);
  assert(getSharedMembers().CaConcentrationConnect->size()==1);
  assert(getSharedMembers().dimensionsConnect);
  assert(getSharedMembers().dimensionsConnect->size()==1);    
  CaConcentration = &((*(getSharedMembers().CaConcentrationConnect))[0]);
  dimension = (*(getSharedMembers().dimensionsConnect))[0];
}

CaERConcentrationJunctionPoint::~CaERConcentrationJunctionPoint() 
{
}

