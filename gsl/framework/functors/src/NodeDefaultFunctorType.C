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

#include "NodeDefaultFunctorType.h"
#include "NodeDefaultFunctor.h"
#include "FunctorType.h"
#include "LensContext.h"
#include <iostream>
#include "DataItem.h"
#include "FunctorDataItem.h"
#include "InstanceFactoryQueriable.h"
#include "DataItemQueriable.h"
#include "FunctorDataItem.h"

NodeDefaultFunctorType::NodeDefaultFunctorType() {}

void NodeDefaultFunctorType::getFunctor(std::unique_ptr<Functor> & r_aptr)
{
   r_aptr.reset(new NodeDefaultFunctor());
}


std::string NodeDefaultFunctorType::getName()
{
   return std::string("NodeDefault");
}


std::string NodeDefaultFunctorType::getDescription()
{
   return std::string(NodeInitializerFunctor::_category);
}


void NodeDefaultFunctorType::getQueriable(
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


Functor* NodeDefaultFunctorType::getFunctor()
{
   Functor* f = new NodeDefaultFunctor;
   _functorList.push_back(f);
   return f;
}


NodeDefaultFunctorType::~NodeDefaultFunctorType()
{
}
