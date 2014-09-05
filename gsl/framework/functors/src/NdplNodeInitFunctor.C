// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "NdplNodeInitFunctor.h"
#include "LensContext.h"
#include "DataItem.h"
#include "FunctorType.h"
#include "NDPair.h"
#include "NodeSet.h"
#include "GridLayerDescriptor.h"
#include "Node.h"
#include "ParameterSet.h"
#include "NodeType.h"
#include "Simulation.h"
#include "NDPair.h"
#include "NDPairListDataItem.h"
#include "FunctorDataItem.h"
#include "LayerDefinitionContext.h"
#include "InstanceFactoryQueriable.h"
#include "DataItemQueriable.h"
#include "FunctorDataItem.h"
#include "SyntaxErrorException.h"

#include <list>

NdplNodeInitFunctor::NdplNodeInitFunctor()
{
}


NdplNodeInitFunctor::NdplNodeInitFunctor(const NdplNodeInitFunctor& csf)
{
   if (csf._functor_ap.get())
      csf._functor_ap->duplicate(_functor_ap);
}


void NdplNodeInitFunctor::duplicate(std::auto_ptr<Functor> &fap) const
{
   fap.reset(new NdplNodeInitFunctor(*this));
}


NdplNodeInitFunctor::~NdplNodeInitFunctor()
{
}


void NdplNodeInitFunctor::doInitialize(LensContext *c, 
				       const std::vector<DataItem*>& args)
{
   if (args.size() != 1) {
      throw SyntaxErrorException(
	 "Improper number of initialization arguments passed to NdplNodeInitFunctor");
   }
   FunctorDataItem* fdi = dynamic_cast<FunctorDataItem*>(args[0]);
   if (fdi == 0) {
      throw SyntaxErrorException(
	 "Dynamic cast of DataItem to FunctorDataItem failed on NdplNodeInitFunctor");
   }
   Functor *functor = fdi->getFunctor();
   if (functor ==0) {
      throw SyntaxErrorException(
	 "Functor provided to NdplNodeInitFunctor is not valid!");
   }
   functor->duplicate(_functor_ap);
}


void NdplNodeInitFunctor::doExecute(LensContext *c, 
				    const std::vector<DataItem*>& args, 
				    std::auto_ptr<DataItem>& rvalue)
{
   NodeSet *nodeset = c->layerContext->nodeset;
   std::auto_ptr<ParameterSet> initPset;
   std::vector<NodeDescriptor*>::iterator nodesIter, nodesEnd;
   std::vector<DataItem*> nullArgs;
   std::auto_ptr<DataItem> rval_ap;
   NDPairList dummy;

   std::vector<GridLayerDescriptor*> const & layers = nodeset->getLayers();
   std::vector<GridLayerDescriptor*>::const_iterator iter = layers.begin();
   std::vector<GridLayerDescriptor*>::const_iterator end = layers.end();
   for (; iter != end; ++iter) {
      GridLayerDescriptor* gld = (*iter);
      std::vector<NodeDescriptor*>  nodes;
      nodeset->getNodes(nodes, gld);
      c->sim->getNodeType(
	 gld->getModelName(), dummy)->getInitializationParameterSet(initPset);
      nodesIter = nodes.begin();
      nodesEnd = nodes.end();
      for (; nodesIter != nodesEnd; ++nodesIter) {
         _functor_ap->execute(c, nullArgs, rval_ap);
         NDPairListDataItem* ndpldi = 
	    dynamic_cast<NDPairListDataItem*>(rval_ap.get());
         if (ndpldi == 0) {
            throw SyntaxErrorException(
	       "Dynamic cast of DataItem to NDPairListDataItem failed on NdplNodeInitFunctor");
         }
         initPset->set(*(ndpldi->getNDPairList()));
	 // @TODO Distributed local filter
         if ((*nodesIter)->getNode())       // added by Jizhu Lu on 12/04/2005
            (*nodesIter)->getNode()->initialize(initPset.get());
      }
   }
}
