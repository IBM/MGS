// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "NdplInAttrInitFunctor.h"
#include "GslContext.h"
#include "DataItem.h"
#include "FunctorType.h"
#include "NDPairList.h"
#include "ConnectionContext.h"
#include "ParameterSet.h"
#include "ParameterSetDataItem.h"
#include "FunctorDataItem.h"
#include "NDPairListDataItem.h"
#include "NodeSet.h"
#include "Node.h"
#include "GridLayerDescriptor.h"
#include "Simulation.h"
#include "InstanceFactoryQueriable.h"
#include "DataItemQueriable.h"
#include "FunctorDataItem.h"
#include "SyntaxErrorException.h"

NdplInAttrInitFunctor::NdplInAttrInitFunctor()
{
}


NdplInAttrInitFunctor::NdplInAttrInitFunctor(const NdplInAttrInitFunctor& csf)
{
   if (csf._functor_ap.get())
      csf._functor_ap->duplicate(std::move(_functor_ap));
}


void NdplInAttrInitFunctor::duplicate(std::unique_ptr<Functor>&& fap) const
{
   fap.reset(new NdplInAttrInitFunctor(*this));
}


NdplInAttrInitFunctor::~NdplInAttrInitFunctor()
{
}


void NdplInAttrInitFunctor::doInitialize(GslContext *c, const std::vector<DataItem*>& args)
{
   if (args.size() != 1) {
      throw SyntaxErrorException(
	 "Improper number of initialization arguments passed to NdplInAttrInitFunctor!");
   }
   FunctorDataItem* fdi = dynamic_cast<FunctorDataItem*>(args[0]);
   if (fdi == 0) {
      throw SyntaxErrorException(
	 "Dynamic cast of DataItem to FunctorDataItem failed in NdplInAttrInitFunctor");
   }
   Functor *functor = fdi->getFunctor();
   if (functor ==0) {
      throw SyntaxErrorException(
	 "Functor provided to NdplInAttrInitFunctor is not valid");
   }
   functor->duplicate(std::move(_functor_ap));
}


void NdplInAttrInitFunctor::doExecute(GslContext *c, 
				      const std::vector<DataItem*>& args, 
				      std::unique_ptr<DataItem>& rvalue)
{
   std::vector<DataItem*> nullArgs;
   std::unique_ptr<DataItem> rval_ap;
   NDPairList dummy;
   std::unique_ptr<ParameterSet> pset;

   ConnectionContext *cc = c->connectionContext;

   c->sim->getNodeType(cc->destinationNode->getGridLayerDescriptor()->getModelName(), 
      dummy)->getInAttrParameterSet(std::move(pset));

   _functor_ap->execute(c, nullArgs, rval_ap);

   NDPairListDataItem* ndpldi = 
      dynamic_cast<NDPairListDataItem*>(rval_ap.get());
   if (ndpldi == 0) {
      throw SyntaxErrorException(
	 "Dynamic cast of DataItem to NDPairListDataItem failed on NdplInAttrInitFunctor");
   }

   pset->set(*(ndpldi->getNDPairList()));

   ParameterSetDataItem *psdi= new ParameterSetDataItem();
   psdi->setParameterSet(pset);
   rvalue.reset(psdi);

}
