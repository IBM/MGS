// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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

void EdgeDefaultFunctor::duplicate(std::unique_ptr<Functor>&& fap) const
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
