#include "Lens.h"
#include "DNNode.h"
#include "CG_DNNode.h"
#include "rndm.h"
#include <cfloat>

#define TOASTED
#ifdef TOASTED
#include "IsToast.h"
#endif

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
  ShallowArray<EdgeSetInput>::iterator iter, end = inputs.end();
  if (!ready) {
    for (iter=inputs.begin(); iter!=end; ++iter) {
      ready = (*iter->inputArray)[iter->inputIndex] != PRELIM_STATE;
      if (!ready) break;
    }
  }
  if (ready) {
    output = 0;
#ifdef TOASTED
    if (isToast(*weightedGradient,getSimulation().getIteration())) assert(0);
#endif
    for (iter=inputs.begin(); iter!=end; ++iter) {
      output += (*iter->inputArray)[iter->inputIndex];
    }
    gradient = *weightedGradient;
#ifdef TOASTED
    if (isToast(output,getSimulation().getIteration())) assert(0);
#endif
  }
}

void DNNode::extractInputIndex(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_DNNodeInAttrPSet* CG_inAttrPset, CG_DNNodeOutAttrPSet* CG_outAttrPset) 
{
  inputs[inputs.size()-1].inputIndex = CG_inAttrPset->index;
}

DNNode::~DNNode() 
{
}

