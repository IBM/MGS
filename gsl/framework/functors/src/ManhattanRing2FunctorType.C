// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "ManhattanRing2FunctorType.h"
#include "ManhattanRing2Functor.h"
#include "FunctorType.h"
#include "GslContext.h"
//#include <iostream>
#include "DataItem.h"
#include "FunctorDataItem.h"
#include "InstanceFactoryQueriable.h"
#include "DataItemQueriable.h"
#include "FunctorDataItem.h"

ManhattanRing2FunctorType::ManhattanRing2FunctorType() {}

void ManhattanRing2FunctorType::getFunctor(std::unique_ptr<Functor> & r_aptr)
{
   r_aptr.reset(new ManhattanRing2Functor());
}


std::string ManhattanRing2FunctorType::getName()
{
   return std::string("ManhattanRing2");
}


std::string ManhattanRing2FunctorType::getDescription()
{
   return std::string(SampFctr1Functor::_category);
}


void ManhattanRing2FunctorType::getQueriable(
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


Functor* ManhattanRing2FunctorType::getFunctor()
{
   Functor* f = new ManhattanRing2Functor;
   _functorList.push_back(f);
   return f;
}


ManhattanRing2FunctorType::~ManhattanRing2FunctorType()
{
}
