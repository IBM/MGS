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

#include "ConnectSets2FunctorType.h"
#include "ConnectSets2Functor.h"
#include "FunctorType.h"
#include "LensContext.h"
//#include <iostream>
#include "DataItem.h"
#include "FunctorDataItem.h"
#include "InstanceFactoryQueriable.h"
#include "DataItemQueriable.h"
#include "FunctorDataItem.h"

ConnectSets2FunctorType::ConnectSets2FunctorType() {}

void ConnectSets2FunctorType::getFunctor(std::auto_ptr<Functor> & r_aptr)
{
   r_aptr.reset(new ConnectSets2Functor());
}


std::string ConnectSets2FunctorType::getName()
{
   return std::string("ConnectSets2");
}


std::string ConnectSets2FunctorType::getDescription()
{
   return std::string(ConnectorFunctor::_category);
}


void ConnectSets2FunctorType::getQueriable(
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


Functor* ConnectSets2FunctorType::getFunctor()
{
   Functor* f = new ConnectSets2Functor;
   _functorList.push_back(f);
   return f;
}


ConnectSets2FunctorType::~ConnectSets2FunctorType()
{
}
