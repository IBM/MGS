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

#include "EachDstPropSrcFunctorType.h"
#include "EachDstPropSrcFunctor.h"
#include "FunctorType.h"
#include "LensContext.h"
//#include <iostream>
#include "DataItem.h"
#include "FunctorDataItem.h"
#include "InstanceFactoryQueriable.h"
#include "DataItemQueriable.h"
#include "FunctorDataItem.h"

EachDstPropSrcFunctorType::EachDstPropSrcFunctorType() {}

void EachDstPropSrcFunctorType::getFunctor(std::auto_ptr<Functor> & r_aptr)
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


Functor* EachDstPropSrcFunctorType::getFunctor()
{
   Functor* f = new EachDstPropSrcFunctor;
   _functorList.push_back(f);
   return f;
}


EachDstPropSrcFunctorType::~EachDstPropSrcFunctorType()
{
}
