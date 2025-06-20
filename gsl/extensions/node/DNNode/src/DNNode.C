// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#include "Mgs.h"
#include "DNNode.h"
#include "CG_DNNode.h"
#include "rndm.h"
#include <cfloat>
#ifdef HAVE_GPU
#include "CG_DNNodeCompCategory.h"
#endif

#ifdef HAVE_GPU

#define output  (_container->um_output[__index__])
#define gradient  (_container->um_gradient[__index__])
#define inputs  (_container->um_inputs[__index__])
#define weightedGradient  (_container->um_weightedGradient[__index__])
#define ready  (_container->um_ready[__index__])
#endif

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
  auto end = inputs.end();
  if (!ready) {
    /* getIndex() and getNodeIndex() return a Grid-instance-based index 
     * getGlobalIndex() return the node instance index (unique across Grids that define the Layer for that NodeType)
     * */
    for (auto iter=inputs.begin(); iter!=end; ++iter) {
      ready = (*iter->inputArray)[iter->inputIndex] != PRELIM_STATE;
      if (!ready) break;
    }
  }
  if (ready) {
    output = 0;
    for (auto iter=inputs.begin(); iter!=end; ++iter)
    {
      // += h_j * w_{ji}  : which is 'weightedOutput' from by all incoming DNEdgeSet
      output += (*iter->inputArray)[iter->inputIndex];    
    }
    gradient = *weightedGradient;

#ifdef TOASTED
    if (isToast(gradient,getSimulation().getIteration())) assert(0);
    if (isToast(output,getSimulation().getIteration())) assert(0);
#endif
  }
}

void DNNode::extractInputIndex(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_DNNodeInAttrPSet* CG_inAttrPset, CG_DNNodeOutAttrPSet* CG_outAttrPset) 
{
  inputs[inputs.size()-1].inputIndex = CG_inAttrPset->index;
}

DNNode::~DNNode() 
{
}

