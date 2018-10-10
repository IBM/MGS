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

#include "UniformDistFunctor.h"
#include "FunctorType.h"
#include "FunctorDataItem.h"
#include "NumericDataItem.h"
#include "LensContext.h"
#include "InstanceFactoryQueriable.h"
//#include <iostream>
#include "FloatDataItem.h"
#include "DataItemQueriable.h"
#include "FunctorDataItem.h"
#include "SyntaxErrorException.h"
#include "rndm.h"
#include "Simulation.h"

class FunctorType;
class Simulation;

//#define RNG ( (random()+0.1)/2147483648.0)

void UniformDistFunctor::doInitialize(LensContext *c, 
				      const std::vector<DataItem*>& args)
{
   // get min linit
   NumericDataItem* ndi = dynamic_cast<NumericDataItem*>(args[0]);
   if (ndi == 0) {
      throw SyntaxErrorException(
	 "Dynamic cast of DataItem to NumericDataItem failed in UniformDistFunctor");
   }
   _minLim = ndi->getFloat();

   // get max linit
   NumericDataItem* ndi2= dynamic_cast<NumericDataItem*>(args[1]);
   if (ndi2 == 0) {
      throw SyntaxErrorException(
	 "Dynamic cast of DataItem to NumericDataItem failed in UniformDistFunctor");
   }
   _maxLim = ndi2->getFloat();

}


void UniformDistFunctor::doExecute(LensContext *c, 
				   const std::vector<DataItem*>& args, 
				   std::unique_ptr<DataItem>& rvalue)
{
   FloatDataItem fdi;

   float genNum = drandom(_minLim, _maxLim,c->sim->getSharedFunctorRandomSeedGenerator());

   fdi.setFloat(genNum);

   rvalue.reset(new FloatDataItem(fdi));
}


void UniformDistFunctor::duplicate(std::unique_ptr<Functor> &fap) const
{
   fap.reset(new UniformDistFunctor(*this));
}


UniformDistFunctor::UniformDistFunctor()
{
}

UniformDistFunctor::~UniformDistFunctor()
{
}
