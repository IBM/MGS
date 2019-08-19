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

#include "ManhattanRing2FunctorType.h"
#include "ManhattanRing2Functor.h"
#include "FunctorType.h"
#include "LensContext.h"
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
