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

#include "FunctorType.h"
#include "FunctorDataItem.h"
#include "Functor.h"
#include "DataItemQueriable.h"
#include "InstanceFactoryQueriable.h"
#include "SyntaxErrorException.h"
#include "NDPairList.h"

FunctorType::FunctorType()
   : InstanceFactory()
{
}

// not necessary for now Array handles itself

// FunctorType::FunctorType(const FunctorType& rv)
//    : InstanceFactory(rv)
// {
//    copyContents(rv);
// }

// FunctorType& FunctorType::operator=(const FunctorType& rv)
// {
//    if (this == &rv) {
//       return *this;
//    }
//    InstanceFactory::operator=(rv);
//    destructContents();
//    copyContents(rv);
//    return *this;
// }

void FunctorType::getInstance(std::unique_ptr<DataItem> & adi, std::vector<DataItem*> const * args, LensContext* c)
{
   FunctorDataItem* fdi = new FunctorDataItem;

   std::unique_ptr<Functor> af;
   getFunctor(af);
   af->initialize(c, *args);
   fdi->setFunctor(af.get());

// Could not figure out why we need to duplicate and add to
// the instances of the factory, so I disabled the code below - sgc   
//   std::unique_ptr<DataItem> apdi;
//   fdi->duplicate(apdi);
//   _instances.push_back(apdi.release());

   adi.reset(fdi);
}

void FunctorType::getInstance(std::unique_ptr<DataItem> & adi, 
			      const NDPairList& ndplist,
			      LensContext* c)
{
   throw SyntaxErrorException(
      "Functors can not be instantiated with Name-DataItem pair lists.");
}

FunctorType::~FunctorType()
{
   // not necessary for now Array handles itself
   // destructContents();
}

std::string FunctorType::getCategory()
{
   return getDescription();
}

// not necessary for now Array handles itself
// void FunctorType::copyContents(const FunctorType& rv)
// {
//    std::list<Functor*>::const_iterator it, end = rv._functorList.end();
//    for (it = rv._functorList.begin(); it!=end; ++it) {
//       std::unique_ptr<Functor> dup;
//       (*it)->duplicate(dup);
//       _functorList.push_back(dup.release());
//    }
// }

// void FunctorType::destructContents()
// {
//    std::list<Functor*>::iterator it, end = _functorList.end();
//    for (it = _functorList.begin(); it!=end; ++it) {
//       delete (*it);
//    }
// }
