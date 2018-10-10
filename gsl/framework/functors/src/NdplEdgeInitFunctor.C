// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "NdplEdgeInitFunctor.h"
#include "LensContext.h"
#include "DataItem.h"
#include "FunctorType.h"
#include "NDPairList.h"
#include "ConnectionContext.h"
#include "ParameterSet.h"
#include "ParameterSetDataItem.h"
#include "FunctorDataItem.h"
#include "NDPairListDataItem.h"
#include "EdgeType.h"
#include "InstanceFactoryQueriable.h"
#include "DataItemQueriable.h"
#include "FunctorDataItem.h"
#include "SyntaxErrorException.h"

NdplEdgeInitFunctor::NdplEdgeInitFunctor()
{
}


NdplEdgeInitFunctor::NdplEdgeInitFunctor(const NdplEdgeInitFunctor& csf)
{
   if (csf._functor_ap.get())
      csf._functor_ap->duplicate(_functor_ap);
}


void NdplEdgeInitFunctor::duplicate(std::unique_ptr<Functor> &fap) const
{
   fap.reset(new NdplEdgeInitFunctor(*this));
}


NdplEdgeInitFunctor::~NdplEdgeInitFunctor()
{
}


void NdplEdgeInitFunctor::doInitialize(LensContext *c, 
				       const std::vector<DataItem*>& args)
{
   if (args.size() != 1) {
      throw SyntaxErrorException(
	 "Improper number of initialization arguments passed to NdplEdgeInitFunctor!");
   }
   FunctorDataItem* fdi = dynamic_cast<FunctorDataItem*>(args[0]);
   if (fdi == 0) {
      throw SyntaxErrorException(
	 "Dynamic cast of DataItem to FunctorDataItem failed in NdplEdgeInitFunctor");
   }
   Functor *functor = fdi->getFunctor();
   if (functor ==0) {
      throw SyntaxErrorException(
	 "Functor provided to NdplEdgeInitFunctor is not valid");
   }
   functor->duplicate(_functor_ap);
}


void NdplEdgeInitFunctor::doExecute(LensContext *c, 
				    const std::vector<DataItem*>& args, 
				    std::unique_ptr<DataItem>& rvalue)
{
   std::vector<DataItem*> nullArgs;
   std::unique_ptr<DataItem> rval_ap;

   ConnectionContext *cc = c->connectionContext;
   std::unique_ptr<ParameterSet> pset;
   cc->edgeType->getInitializationParameterSet(pset);

   _functor_ap->execute(c, nullArgs, rval_ap);

   NDPairListDataItem* ndpldi = 
      dynamic_cast<NDPairListDataItem*>(rval_ap.get());
   if (ndpldi == 0) {
      throw SyntaxErrorException(
	 "Dynamic cast of DataItem to NDPairListDataItem failed on NdplEdgeInitFunctor");
   }

   pset->set(*(ndpldi->getNDPairList()));

   ParameterSetDataItem *psdi= new ParameterSetDataItem();
   psdi->setParameterSet(pset);
   rvalue.reset(psdi);

}
