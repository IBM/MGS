// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "EachDstPropSrcFunctorType.h"
#include "EachDstPropSrcFunctor.h"
#include "FunctorType.h"
#include "GslContext.h"
//#include <iostream>
#include "DataItem.h"
#include "FunctorDataItem.h"
#include "InstanceFactoryQueriable.h"
#include "DataItemQueriable.h"
#include "FunctorDataItem.h"

EachDstPropSrcFunctorType::EachDstPropSrcFunctorType() {}

void EachDstPropSrcFunctorType::getFunctor(std::unique_ptr<Functor> & r_aptr)
{
   r_aptr.reset(new EachDstPropSrcFunctor());
}


std::string EachDstPropSrcFunctorType::getName()
{
   return std::string("EachDstPropSrc");
}


std::string EachDstPropSrcFunctorType::getDescription()
{
   return std::string(SampFctr2Functor::_category);
}


void EachDstPropSrcFunctorType::getQueriable(
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


Functor* EachDstPropSrcFunctorType::getFunctor()
{
   Functor* f = new EachDstPropSrcFunctor;
   _functorList.push_back(f);
   return f;
}


EachDstPropSrcFunctorType::~EachDstPropSrcFunctorType()
{
}
