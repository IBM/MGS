// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "TraverseFunctor.h"
#include "LensContext.h"
#include "DataItem.h"
#include "FunctorType.h"
#include "InstanceFactoryQueriable.h"
#include "DataItemQueriable.h"
#include "FunctorDataItem.h"

TraverseFunctor::TraverseFunctor()
{
}


void TraverseFunctor::duplicate(std::unique_ptr<Functor>&& fap) const
{
   fap.reset(new TraverseFunctor(*this));
}

TraverseFunctor::~TraverseFunctor()
{
}


void TraverseFunctor::doInitialize(LensContext *c, 
				   const std::vector<DataItem*>& args)
{
}


void TraverseFunctor::doExecute(LensContext *c, 
				const std::vector<DataItem*>& args, 
				std::unique_ptr<DataItem>& rvalue)
{
}
