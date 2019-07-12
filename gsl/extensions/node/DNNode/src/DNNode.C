#include "Lens.h"
#include "DNNode.h"
#include "CG_DNNode.h"
#include "rndm.h"
#include <cfloat>

<<<<<<< HEAD
#define TOASTED
#ifdef TOASTED
#include "IsToast.h"
#endif

=======
>>>>>>> Adding DNN model suite.
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
<<<<<<< HEAD
  ShallowArray<EdgeSetInput>::iterator iter, end = inputs.end();
  if (!ready) {
    for (iter=inputs.begin(); iter!=end; ++iter) {
      ready = (*iter->inputArray)[iter->inputIndex] != PRELIM_STATE;
=======
  ShallowArray<double*>::iterator iter, end = inputs.end();
  if (!ready) {
    for (iter=inputs.begin(); iter!=end; ++iter) {
      ready = **iter != PRELIM_STATE;
>>>>>>> Adding DNN model suite.
      if (!ready) break;
    }
  }
  if (ready) {
    output = 0;
<<<<<<< HEAD
    for (iter=inputs.begin(); iter!=end; ++iter)
      output += (*iter->inputArray)[iter->inputIndex];    
    gradient = *weightedGradient;

#ifdef TOASTED
    if (isToast(gradient,getSimulation().getIteration())) assert(0);
    if (isToast(output,getSimulation().getIteration())) assert(0);
#endif
  }
}

void DNNode::extractInputIndex(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_DNNodeInAttrPSet* CG_inAttrPset, CG_DNNodeOutAttrPSet* CG_outAttrPset) 
{
  inputs[inputs.size()-1].inputIndex = CG_inAttrPset->index;
=======
    for (iter=inputs.begin(); iter!=end; ++iter) output += **iter;				   
    gradient = *weightedGradient;
  }
}

void DNNode::extractInputPointer(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_DNNodeInAttrPSet* CG_inAttrPset, CG_DNNodeOutAttrPSet* CG_outAttrPset) 
{
  if (CG_inAttrPset->index > inputs.size()-1) inputs.increaseSizeTo(CG_inAttrPset->index+1);
  inputs.push_back(&((*SHD.currentConnection)[CG_inAttrPset->index]));
>>>>>>> Adding DNN model suite.
}

DNNode::~DNNode() 
{
}

