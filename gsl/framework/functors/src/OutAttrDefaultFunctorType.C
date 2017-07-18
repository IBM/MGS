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

#include "OutAttrDefaultFunctorType.h"
#include "OutAttrDefaultFunctor.h"
#include "FunctorType.h"
#include "LensContext.h"
//#include <iostream>
#include "DataItem.h"
#include "FunctorDataItem.h"
#include "InstanceFactoryQueriable.h"
#include "DataItemQueriable.h"
#include "FunctorDataItem.h"

OutAttrDefaultFunctorType::OutAttrDefaultFunctorType() {}

void OutAttrDefaultFunctorType::getFunctor(std::auto_ptr<Functor> & r_aptr)
{
   r_aptr.reset(new OutAttrDefaultFunctor());
}


std::string OutAttrDefaultFunctorType::getName()
{
   return std::string("OutAttrDefault");
}


std::string OutAttrDefaultFunctorType::getDescription()
{
   return std::string(Functor::_category);
}


void OutAttrDefaultFunctorType::getQueriable(
   std::auto_ptr<InstanceFactoryQueriable>& dup)
{
   dup.reset(new InstanceFactoryQueriable(this));
   Array<Functor*>::iterator it, end = _functorList.end();
   for(it = _functorList.begin(); it!=end; ++it) {
      Functor* f = (*it);
      FunctorDataItem* fdi = new FunctorDataItem;
      fdi->setFunctor(f);
      std::auto_ptr<DataItem> apdi(fdi);
      DataItemQueriable* diq = new DataItemQueriable(apdi);
      diq->setName(getName());
      diq->setDescription(getDescription());
      std::auto_ptr<DataItemQueriable> apq(diq);
      dup->addQueriable(apq);
   }
   dup->setName(getName());
}


Functor* OutAttrDefaultFunctorType::getFunctor()
{
   Functor* f = new OutAttrDefaultFunctor;
   _functorList.push_back(f);
   return f;
}


OutAttrDefaultFunctorType::~OutAttrDefaultFunctorType()
{
}
