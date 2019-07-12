#include "Lens.h"
#include "DNNode.h"
#include "CG_DNNode.h"
#include "rndm.h"
#include <cfloat>

#define PRELIM_STATE DBL_MAX
#define SHD getSharedMembers()

void DNNode::initialize(RNG& rng) 
{
  output = PRELIM_STATE;
  gradient = PRELIM_STATE;
  ready = false;
}

void DNNode::update(RNG& rng) 
{
  ShallowArray<double*>::iterator iter, end = inputs.end();
  if (!ready) {
    for (iter=inputs.begin(); iter!=end; ++iter) {
      ready = **iter != PRELIM_STATE;
      if (!ready) break;
    }
  }
  if (ready) {
    output = 0;
    for (iter=inputs.begin(); iter!=end; ++iter) output += **iter;				   
    gradient = *weightedGradient;
  }
}

void DNNode::extractInputPointer(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_DNNodeInAttrPSet* CG_inAttrPset, CG_DNNodeOutAttrPSet* CG_outAttrPset) 
{
  if (CG_inAttrPset->index > inputs.size()-1) inputs.increaseSizeTo(CG_inAttrPset->index+1);
  inputs.push_back(&((*SHD.currentConnection)[CG_inAttrPset->index]));
}

DNNode::~DNNode() 
{
}

