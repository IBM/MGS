// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "SameFunctor.h"
#include "GslContext.h"
#include "DataItem.h"
#include "FunctorType.h"
#include "ParameterSetDataItem.h"
#include "NodeSet.h"
#include "LayerDefinitionContext.h"
#include "Node.h"
#include "ParameterSet.h"
#include "InstanceFactoryQueriable.h"
#include "DataItemQueriable.h"
#include "FunctorDataItem.h"
#include "SyntaxErrorException.h"

SameFunctor::SameFunctor()
{
}


SameFunctor::SameFunctor(const SameFunctor& csf)
{
   if (csf._pset.get()) csf._pset.get()->duplicate(std::move(_pset));
}


void SameFunctor::duplicate(std::unique_ptr<Functor>&& fap) const
{
   fap.reset(new SameFunctor(*this));
}


SameFunctor::~SameFunctor()
{

}


void SameFunctor::doInitialize(GslContext *c, 
			       const std::vector<DataItem*>& args)
{
   if (args.size() != 1) {
      throw SyntaxErrorException(
	 "Improper number of initialization arguments passed to SameFunctor");
   }
   ParameterSetDataItem* psdi = dynamic_cast<ParameterSetDataItem*>(args[0]);
   if (psdi == 0) {
      throw SyntaxErrorException(
	 "Dynamic cast of DataItem to ParameterSetDataItem failed on SameFunctor");
   }
   if (psdi->getParameterSet()) psdi->getParameterSet()->duplicate(std::move(_pset));
   else {
      throw SyntaxErrorException(
	 "Bad ParameterSetDataItem passed to initialize SameFunctor");
   }
}

void SameFunctor::doExecute(GslContext *c, 
			    const std::vector<DataItem*>& args, 
			    std::unique_ptr<DataItem>& rvalue)
{
   NodeSet *nodeset = c->layerContext->nodeset;
   std::vector<NodeDescriptor*>::iterator nodesIter, nodesEnd;
   std::vector<GridLayerDescriptor*> const & layers = nodeset->getLayers();
   std::vector<GridLayerDescriptor*>::const_iterator iter = layers.begin();
   std::vector<GridLayerDescriptor*>::const_iterator end = layers.end();
   for (; iter != end; ++iter) {
      GridLayerDescriptor* gld = (*iter);
      std::vector<NodeDescriptor*>  nodes;
      nodeset->getNodes(nodes, gld);
      nodesIter = nodes.begin();
      nodesEnd = nodes.end();
      for (; nodesIter != nodesEnd; ++nodesIter) {
       	 // @TODO Distributed local filter
         if ((*nodesIter)->getNode())    // added by Jizhu Lu on 12/04/2005
	    (*nodesIter)->getNode()->initialize(_pset.get());
      }
   }
}
