// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "ManhattanRingFunctorType.h"
#include "ManhattanRingFunctor.h"
#include "FunctorType.h"
#include "GslContext.h"
//#include <iostream>
#include "DataItem.h"
#include "FunctorDataItem.h"
#include "InstanceFactoryQueriable.h"
#include "DataItemQueriable.h"
#include "FunctorDataItem.h"

ManhattanRingFunctorType::ManhattanRingFunctorType() {}

void ManhattanRingFunctorType::getFunctor(std::unique_ptr<Functor> & r_aptr)
{
   r_aptr.reset(new ManhattanRingFunctor());
}


std::string ManhattanRingFunctorType::getName()
{
   return std::string("ManhattanRing");
}


std::string ManhattanRingFunctorType::getDescription()
{
   return std::string(SampFctr1Functor::_category);
}


void ManhattanRingFunctorType::getQueriable(
   std::unique_ptr<InstanceFactoryQueriable>& dup)
{
   dup.reset(new InstanceFactoryQueriable(this));
   Array<Functor*>::iterator it, end = _functorList.end();
   for(it = _functorList.begin(); it!=end; ++it) {
      Functor* f = (*it);
      FunctorDataItem* fdi = new FunctorDataItem;
      fdi->setFunctor(f);
      std::unique_ptr<DataItem> apdi(fdi);
      DataItemQueriable* diq = new DataItemQueriable(apdi);
      diq->setName(getName());
      diq->setDescription(getDescription());
      std::unique_ptr<DataItemQueriable> apq(diq);
      dup->addQueriable(apq);
   }
   dup->setName(getName());
}


Functor* ManhattanRingFunctorType::getFunctor()
{
   Functor* f = new ManhattanRingFunctor;
   _functorList.push_back(f);
   return f;
}


ManhattanRingFunctorType::~ManhattanRingFunctorType()
{
}
