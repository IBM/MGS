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

#include "NdplInAttrInitFunctor.h"
#include "LensContext.h"
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
      csf._functor_ap->duplicate(_functor_ap);
}


void NdplInAttrInitFunctor::duplicate(std::auto_ptr<Functor> &fap) const
{
   fap.reset(new NdplInAttrInitFunctor(*this));
}


NdplInAttrInitFunctor::~NdplInAttrInitFunctor()
{
}


void NdplInAttrInitFunctor::doInitialize(LensContext *c, const std::vector<DataItem*>& args)
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
   functor->duplicate(_functor_ap);
}


void NdplInAttrInitFunctor::doExecute(LensContext *c, 
				      const std::vector<DataItem*>& args, 
				      std::auto_ptr<DataItem>& rvalue)
{
   std::vector<DataItem*> nullArgs;
   std::auto_ptr<DataItem> rval_ap;
   NDPairList dummy;
   std::auto_ptr<ParameterSet> pset;

   ConnectionContext *cc = c->connectionContext;

   c->sim->getNodeType(cc->destinationNode->getGridLayerDescriptor()->getModelName(), 
      dummy)->getInAttrParameterSet(pset);

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
