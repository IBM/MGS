// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "NdplInAttrInitFunctorType.h"
#include "NdplInAttrInitFunctor.h"
#include "FunctorType.h"
#include "LensContext.h"
//#include <iostream>
#include "DataItem.h"
#include "FunctorDataItem.h"
#include "InstanceFactoryQueriable.h"
#include "DataItemQueriable.h"
#include "FunctorDataItem.h"

NdplInAttrInitFunctorType::NdplInAttrInitFunctorType() {}

void NdplInAttrInitFunctorType::getFunctor(std::auto_ptr<Functor> & r_aptr)
{
   r_aptr.reset(new NdplInAttrInitFunctor());
}


std::string NdplInAttrInitFunctorType::getName()
{
   return std::string("NdplInAttrInit");
}


std::string NdplInAttrInitFunctorType::getDescription()
{
   return std::string(InAttrInitializerFunctor::_category);
}


void NdplInAttrInitFunctorType::getQueriable(
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


Functor* NdplInAttrInitFunctorType::getFunctor()
{
   Functor* f = new NdplInAttrInitFunctor;
   _functorList.push_back(f);
   return f;
}


NdplInAttrInitFunctorType::~NdplInAttrInitFunctorType()
{
}
