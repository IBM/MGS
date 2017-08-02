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

#include "EdgeDefaultFunctorType.h"
#include "EdgeDefaultFunctor.h"
#include "FunctorType.h"
#include "LensContext.h"
//#include <iostream>
#include "DataItem.h"
#include "FunctorDataItem.h"
#include "InstanceFactoryQueriable.h"
#include "DataItemQueriable.h"
#include "FunctorDataItem.h"

EdgeDefaultFunctorType::EdgeDefaultFunctorType() {}

void EdgeDefaultFunctorType::getFunctor(std::auto_ptr<Functor> & r_aptr)
{
   r_aptr.reset(new EdgeDefaultFunctor());
}


std::string EdgeDefaultFunctorType::getName()
{
   return std::string("EdgeDefault");
}


std::string EdgeDefaultFunctorType::getDescription()
{
   return std::string(EdgeInitializerFunctor::_category);
}


void EdgeDefaultFunctorType::getQueriable(
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


Functor* EdgeDefaultFunctorType::getFunctor()
{
   Functor* f = new EdgeDefaultFunctor;
   _functorList.push_back(f);
   return f;
}


EdgeDefaultFunctorType::~EdgeDefaultFunctorType()
{
}
