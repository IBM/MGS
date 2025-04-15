// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "NodeDefaultFunctor.h"
#include "LensContext.h"
#include "DataItem.h"
#include "FunctorType.h"
#include "NodeSet.h"
#include "GridLayerDescriptor.h"
#include "Node.h"
#include "ParameterSet.h"
#include "NodeType.h"
#include "Simulation.h"
#include "LayerDefinitionContext.h"
#include "InstanceFactoryQueriable.h"
#include "DataItemQueriable.h"
#include "FunctorDataItem.h"

NodeDefaultFunctor::NodeDefaultFunctor()
{
}

void NodeDefaultFunctor::duplicate(std::unique_ptr<Functor>&& fap) const
{
   fap.reset(new NodeDefaultFunctor(*this));
}


NodeDefaultFunctor::~NodeDefaultFunctor()
{
}


void NodeDefaultFunctor::doInitialize(LensContext *c, 
				      const std::vector<DataItem*>& args)
{
}


void NodeDefaultFunctor::doExecute(LensContext *c, 
				   const std::vector<DataItem*>& args, 
				   std::unique_ptr<DataItem>& rvalue)
{
   std::unique_ptr<ParameterSet> initPset;
   std::vector<NodeDescriptor*>  nodes;
   std::vector<NodeDescriptor*>::iterator node, nodesEnd;

   NodeSet *nodeset = c->layerContext->nodeset;
   std::vector<GridLayerDescriptor*> const & layers = nodeset->getLayers();
   std::vector<GridLayerDescriptor*>::const_iterator gld = layers.begin();
   std::vector<GridLayerDescriptor*>::const_iterator end = layers.end();
   for (; gld != end; ++gld) {
      (*gld)->getNodeType()->getInitializationParameterSet(std::move(initPset));

      nodes.clear();
      nodeset->getNodes(nodes, *gld);
      node = nodes.begin();
      nodesEnd = nodes.end();
      for (; node != nodesEnd; ++node) {
	 // @TODO Distributed local filter
         if ((*node)->getNode())    // added by Jizhu Lu on 12/04/2005
            (*node)->getNode()->initialize(initPset.get());
      }
   }
}
