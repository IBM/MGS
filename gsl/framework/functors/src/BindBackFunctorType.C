// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "BindBackFunctorType.h"
#include "BindBackFunctor.h"
#include "FunctorType.h"
#include "LensContext.h"
//#include <iostream>
#include "DataItem.h"
#include "FunctorDataItem.h"
#include "InstanceFactoryQueriable.h"
#include "DataItemQueriable.h"
#include "FunctorDataItem.h"

BindBackFunctorType::BindBackFunctorType() {}

void BindBackFunctorType::getFunctor(std::auto_ptr<Functor> & r_aptr)
{
   r_aptr.reset(new BindBackFunctor());
}


std::string BindBackFunctorType::getName()
{
   return std::string("BindBack");
}


std::string BindBackFunctorType::getDescription()
{
   return std::string(Functor::_category);
}


void BindBackFunctorType::getQueriable(
   std::auto_ptr<InstanceFactoryQueriable>& dup)
{
   dup.reset(new InstanceFactoryQueriable(this));
   Array<Functor*>::iterator it, end = _functorList.end();
   for (it = _functorList.begin(); it!=end; ++it) {
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


Functor* BindBackFunctorType::getFunctor()
{
   Functor* f = new BindBackFunctor;
   _functorList.push_back(f);
   return f;
}


BindBackFunctorType::~BindBackFunctorType()
{
}
