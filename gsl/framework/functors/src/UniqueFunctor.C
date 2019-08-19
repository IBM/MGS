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

#include "UniqueFunctor.h"
#include "LensContext.h"
#include "DataItem.h"
#include "FunctorType.h"
#include "Node.h"
#include "FunctorDataItem.h"
#include "NodePairDataItem.h"
#include "ConnectionContext.h"
#include "InstanceFactoryQueriable.h"
#include "DataItemQueriable.h"
#include "FunctorDataItem.h"
#include "SyntaxErrorException.h"

UniqueFunctor::UniqueFunctor()
{
}


UniqueFunctor::UniqueFunctor(const UniqueFunctor& csf)
{
   if (csf._functor_ap.get())
      csf._functor_ap->duplicate(_functor_ap);
}


void UniqueFunctor::duplicate(std::unique_ptr<Functor> &fap) const
{
   fap.reset(new UniqueFunctor(*this));
}


UniqueFunctor::~UniqueFunctor()
{
}


void UniqueFunctor::doInitialize(LensContext *c, 
				 const std::vector<DataItem*>& args)
{
   // prototype SamplingFctr2 unique(SamplingFctr2 sf);
   // should get a functor returning nodes

   FunctorDataItem* fdi = dynamic_cast<FunctorDataItem*>(args[0]);
   if (fdi == 0) {
      throw SyntaxErrorException(
	 "Argument of UniqueFunctor is not a FunctorDataItem");
   }
   else {

      Functor *functor = fdi->getFunctor();
      if ( functor == 0 ) {
         throw SyntaxErrorException(
	    "FunctorDataItem in UniqueFunctor doesn't hold a proper functor");
      }
      functor->duplicate(_functor_ap);
   }
}


void UniqueFunctor::doExecute(LensContext *c, 
			      const std::vector<DataItem*>& args, 
			      std::unique_ptr<DataItem>& rvalue)
{
   std::vector<DataItem*> nullArgs;
   std::unique_ptr<DataItem> rval_ap;

   ConnectionContext *cc = c->connectionContext;

   while(true) {
      _functor_ap->execute(c, nullArgs, rval_ap);
      if(cc->sourceNode != cc->destinationNode 
	 || cc->sourceNode ==0 || cc->destinationNode==0) break;
      cc->restart = false;
   }
}
