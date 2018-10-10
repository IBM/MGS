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

#include "EdgeDefaultFunctor.h"
#include "LensContext.h"
#include "DataItem.h"
#include "FunctorType.h"
#include "ConnectionContext.h"
#include "ParameterSet.h"
#include "ParameterSetDataItem.h"
#include "EdgeType.h"
#include "InstanceFactoryQueriable.h"
#include "DataItemQueriable.h"
#include "FunctorDataItem.h"

EdgeDefaultFunctor::EdgeDefaultFunctor()
{
}

void EdgeDefaultFunctor::duplicate(std::unique_ptr<Functor> &fap) const
{
   fap.reset(new EdgeDefaultFunctor(*this));
}


EdgeDefaultFunctor::~EdgeDefaultFunctor()
{
}


void EdgeDefaultFunctor::doInitialize(LensContext *c, 
				      const std::vector<DataItem*>& args)
{
}


void EdgeDefaultFunctor::doExecute(LensContext *c, 
				   const std::vector<DataItem*>& args, 
				   std::unique_ptr<DataItem>& rvalue)
{

   ConnectionContext *cc = c->connectionContext;

   std::unique_ptr<ParameterSet> pset;
   cc->edgeType->getInitializationParameterSet(pset);

   ParameterSetDataItem *psdi= new ParameterSetDataItem();
   psdi->setParameterSet(pset);
   rvalue.reset(psdi);
}
