// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "SameFunctorType.h"
#include "SameFunctor.h"
#include "FunctorType.h"
#include "GslContext.h"
//#include <iostream>
#include "DataItem.h"
#include "FunctorDataItem.h"
#include "InstanceFactoryQueriable.h"
#include "DataItemQueriable.h"
#include "FunctorDataItem.h"

SameFunctorType::SameFunctorType() {}

void SameFunctorType::getFunctor(std::unique_ptr<Functor> & r_aptr)
{
   r_aptr.reset(new SameFunctor());
}


std::string SameFunctorType::getName()
{
   return std::string("Same");
}


std::string SameFunctorType::getDescription()
{
   return std::string(NodeInitializerFunctor::_category);
}


void SameFunctorType::getQueriable(
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


Functor* SameFunctorType::getFunctor()
{
   Functor* f = new SameFunctor;
   _functorList.push_back(f);
   return f;
}


SameFunctorType::~SameFunctorType()
{
}
