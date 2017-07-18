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

#include "GaussianFunctor.h"
#include "FunctorType.h"
#include "FunctorDataItem.h"
#include "NumericDataItem.h"
#include "LensContext.h"
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

void GaussianFunctor::doInitialize(LensContext *c, 
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

void GaussianFunctor::doExecute(LensContext *c, 
				const std::vector<DataItem*>& args, 
				std::auto_ptr<DataItem>& rvalue)
{
   FloatDataItem fdi;
   fdi.setFloat(gaussian(_mean,_stddev,c->sim->getSharedFunctorRandomSeedGenerator()));

   rvalue.reset(new FloatDataItem(fdi));
}


void GaussianFunctor::duplicate(std::auto_ptr<Functor> &fap) const
{
   fap.reset(new GaussianFunctor(*this));
}


GaussianFunctor::GaussianFunctor()
{
}

GaussianFunctor::~GaussianFunctor()
{
}
