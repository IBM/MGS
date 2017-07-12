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

#include "UniformLayoutFunctor.h"
#include "FunctorType.h"
#include "NumericDataItem.h"
#include "LensContext.h"
#include "InstanceFactoryQueriable.h"
#include "DataItemQueriable.h"
#include "FunctorDataItem.h"
//#include <iostream>
#include "SyntaxErrorException.h"
class FunctorType;
class Simulation;

void UniformLayoutFunctor::doInitialize(LensContext *c, 
					const std::vector<DataItem*>& args)
{
   NumericDataItem* densityDI = 
      dynamic_cast<NumericDataItem*>(*(args.begin()));
   if (densityDI == 0) {
      throw SyntaxErrorException(
	 "Dynamic cast of DataItem to NumericDataItem failed in UniformLayoutFunctor");
   }
   int density;
   density = densityDI->getInt();

   std::vector<int> &dv = *_density.getModifiableIntVector();
   dv.clear();
   dv.push_back(density);
}


void UniformLayoutFunctor::doExecute(LensContext *c, 
				     const std::vector<DataItem*>& args, 
				     std::auto_ptr<DataItem>& rvalue)
{
   rvalue.reset(new IntArrayDataItem(_density));
}


void UniformLayoutFunctor::duplicate(std::auto_ptr<Functor> &fap) const
{
   fap.reset(new UniformLayoutFunctor(*this));
}


UniformLayoutFunctor::UniformLayoutFunctor()
{
}

UniformLayoutFunctor::~UniformLayoutFunctor()
{
}
