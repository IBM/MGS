// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

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

void NodeDefaultFunctor::duplicate(std::auto_ptr<Functor> &fap) const
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
				   std::auto_ptr<DataItem>& rvalue)
{
   std::auto_ptr<ParameterSet> initPset;
   std::vector<NodeDescriptor*>  nodes;
   std::vector<NodeDescriptor*>::iterator node, nodesEnd;

   NodeSet *nodeset = c->layerContext->nodeset;
   std::vector<GridLayerDescriptor*> const & layers = nodeset->getLayers();
   std::vector<GridLayerDescriptor*>::const_iterator gld = layers.begin();
   std::vector<GridLayerDescriptor*>::const_iterator end = layers.end();
   for (; gld != end; ++gld) {
      (*gld)->getNodeType()->getInitializationParameterSet(initPset);

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
