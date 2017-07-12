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


void TraverseFunctor::duplicate(std::auto_ptr<Functor> &fap) const
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
				std::auto_ptr<DataItem>& rvalue)
{
}
