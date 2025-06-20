// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "UniformLayoutFunctor.h"
#include "FunctorType.h"
#include "NumericDataItem.h"
#include "GslContext.h"
#include "InstanceFactoryQueriable.h"
#include "DataItemQueriable.h"
#include "FunctorDataItem.h"
//#include <iostream>
#include "SyntaxErrorException.h"
class FunctorType;
class Simulation;

void UniformLayoutFunctor::doInitialize(GslContext *c, 
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


void UniformLayoutFunctor::doExecute(GslContext *c, 
				     const std::vector<DataItem*>& args, 
				     std::unique_ptr<DataItem>& rvalue)
{
   rvalue.reset(new IntArrayDataItem(_density));
}


void UniformLayoutFunctor::duplicate(std::unique_ptr<Functor>&& fap) const
{
   fap.reset(new UniformLayoutFunctor(*this));
}


UniformLayoutFunctor::UniformLayoutFunctor()
{
}

UniformLayoutFunctor::~UniformLayoutFunctor()
{
}
