// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "GaussianFunctor.h"
#include "FunctorType.h"
#include "FunctorDataItem.h"
#include "NumericDataItem.h"
#include "GslContext.h"
#include "InstanceFactoryQueriable.h"
//#include <iostream>
#include "rndm.h"
//#include "Gaussian.h"
#include "FloatDataItem.h"
#include "DataItemQueriable.h"
#include "FunctorDataItem.h"
#include "SyntaxErrorException.h"
#include "Simulation.h"
class FunctorType;
class Simulation;

void GaussianFunctor::doInitialize(GslContext *c, 
				   const std::vector<DataItem*>& args)
{
   // get mean and stddev
   NumericDataItem* ndi = dynamic_cast<NumericDataItem*>(args[0]);
   if (ndi == 0) {
      throw SyntaxErrorException(
	 "Dynamic cast of DataItem to NumericDataItem failed in GaussianFunctor");
   }
   _mean = ndi->getFloat();
   ndi = dynamic_cast<NumericDataItem*>(args[1]);
   if (ndi == 0) {
      throw SyntaxErrorException(
	 "Dynamic cast of DataItem to NumericDataItem failed in GaussianFunctor");
   }
   _stddev = ndi->getFloat();

}

void GaussianFunctor::doExecute(GslContext *c, 
				const std::vector<DataItem*>& args, 
				std::unique_ptr<DataItem>& rvalue)
{
   FloatDataItem fdi;
   fdi.setFloat(gaussian(_mean,_stddev,c->sim->getSharedFunctorRandomSeedGenerator()));

   rvalue.reset(new FloatDataItem(fdi));
}


void GaussianFunctor::duplicate(std::unique_ptr<Functor>&& fap) const
{
   fap.reset(new GaussianFunctor(*this));
}


GaussianFunctor::GaussianFunctor()
{
}

GaussianFunctor::~GaussianFunctor()
{
}
