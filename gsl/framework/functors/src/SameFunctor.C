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

#include "SameFunctor.h"
#include "LensContext.h"
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
   if (csf._pset.get()) csf._pset.get()->duplicate(_pset);
}


void SameFunctor::duplicate(std::auto_ptr<Functor> &fap) const
{
   fap.reset(new SameFunctor(*this));
}


SameFunctor::~SameFunctor()
{

}


void SameFunctor::doInitialize(LensContext *c, 
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
   if (psdi->getParameterSet()) psdi->getParameterSet()->duplicate(_pset);
   else {
      throw SyntaxErrorException(
	 "Bad ParameterSetDataItem passed to initialize SameFunctor");
   }
}

void SameFunctor::doExecute(LensContext *c, 
			    const std::vector<DataItem*>& args, 
			    std::auto_ptr<DataItem>& rvalue)
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
