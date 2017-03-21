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

#include "NdplModifierFunctor.h"
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

NdplModifierFunctor::NdplModifierFunctor()
   : _ndpList(0)
{
}


NdplModifierFunctor::NdplModifierFunctor(const NdplModifierFunctor& csf)
   : _ndpList(0)
{
   if (csf._functor_ap.get())
      csf._functor_ap->duplicate(_functor_ap);
   if (csf._ndpList) {
      _ndpList = new NDPairList(*csf._ndpList);
   }
}


void NdplModifierFunctor::duplicate(std::auto_ptr<Functor> &fap) const
{
   fap.reset(new NdplModifierFunctor(*this));
}


NdplModifierFunctor::~NdplModifierFunctor()
{
   delete _ndpList;
}


void NdplModifierFunctor::doInitialize(LensContext *c, 
				       const std::vector<DataItem*>& args)
{
   if (args.size() != 2) {
      throw SyntaxErrorException(
	 "Improper number of initialization arguments passed to NdplModifierFunctor");
   }
   FunctorDataItem* fdi = dynamic_cast<FunctorDataItem*>(args[0]);
   if (fdi == 0) {
      throw SyntaxErrorException(
	 "Dynamic cast of DataItem to FunctorDataItem failed in NdplModifierFunctor");
   }
   Functor *functor = fdi->getFunctor();
   if (functor ==0) {
      throw SyntaxErrorException(
	 "Functor provided to NdplModifierFunctor is not valid");
   }
   functor->duplicate(_functor_ap);

   // Now the name value std::pair list
   NDPairListDataItem* ndpldi = dynamic_cast<NDPairListDataItem*>(args[1]);
   if (ndpldi == 0) {
      throw SyntaxErrorException(
	 "Dynamic cast of DataItem to NDPairListDataItem failed on NdplModifierFunctor");
   }
   delete _ndpList;
   _ndpList = new NDPairList;
   NDPairList* diList = ndpldi->getNDPairList();
   NDPairList::const_iterator it, end = diList->end();
   for (it = diList->begin(); it != end; it++) {
      _ndpList->push_back(new NDPair(**it));
   }
}


void NdplModifierFunctor::doExecute(LensContext *c, 
				    const std::vector<DataItem*>& args, 
				    std::auto_ptr<DataItem>& rvalue)
{
   std::vector<DataItem*> nullArgs;
   std::auto_ptr<DataItem> rval;

   _functor_ap->execute(c, nullArgs, rval);
   ParameterSetDataItem *psdi = 
      dynamic_cast<ParameterSetDataItem*>(rval.release());
   if (psdi==0) {
      throw SyntaxErrorException(
	 "NdplModifier: functor did not return a Parameter Set");
   }
   psdi->getParameterSet()->set(*_ndpList);

   rvalue.reset(psdi);

}
