#include "Lens.h"
#include "CaERConcentrationEndPoint.h"
#include "CG_CaERConcentrationEndPoint.h"
#include "rndm.h"

void CaERConcentrationEndPoint::produceInitialState(RNG& rng) 
{
}

void CaERConcentrationEndPoint::produceSolvedCaConcentration(RNG& rng) 
{
}

void CaERConcentrationEndPoint::produceFinishedCaConcentration(RNG& rng) 
{
}

void CaERConcentrationEndPoint::setPointers(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CaERConcentrationEndPointInAttrPSet* CG_inAttrPset, CG_CaERConcentrationEndPointOutAttrPSet* CG_outAttrPset) 
{
  if (CG_inAttrPset->identifier=="distalEnd") {
    assert(getSharedMembers().CaConcentrationConnect);
    assert(getSharedMembers().CaConcentrationConnect->size()>0);    
    CaConcentration = &((*(getSharedMembers().CaConcentrationConnect))[0]);
    assert(getSharedMembers().dimensionsConnect);
    assert(getSharedMembers().dimensionsConnect->size()>0);    
    dimension = (*(getSharedMembers().dimensionsConnect))[0];    
  }
  else if (CG_inAttrPset->identifier=="proximalEnd") {
    assert(getSharedMembers().CaConcentrationConnect);
    assert(getSharedMembers().CaConcentrationConnect->size()>0);    
    CaConcentration = &((*(getSharedMembers().CaConcentrationConnect))[getSharedMembers().CaConcentrationConnect->size()-1]);
    assert(getSharedMembers().dimensionsConnect);
    assert(getSharedMembers().dimensionsConnect->size()>0);    
    dimension = (*(getSharedMembers().dimensionsConnect))[getSharedMembers().dimensionsConnect->size()-1];
  }
  else {
    std::cerr<<"Warning! Unrecognized input identifier on CaConcentrationEndPoint!"<<std::endl;
  }
}

CaERConcentrationEndPoint::~CaERConcentrationEndPoint() 
{
}

