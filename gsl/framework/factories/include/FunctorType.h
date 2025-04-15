// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef FUNCTORTYPE_H
#define FUNCTORTYPE_H
#include "Copyright.h"

#include "InstanceFactory.h"
#include "Functor.h"
#include "DuplicatePointerArray.h"

#include <memory>
#include <vector>
#include <string>

//class Functor;
class DataItem;
class InstanceFactoryQueriable;
class NDPairList;

class FunctorType : public InstanceFactory
{
   public:
      FunctorType();
      // not necessary for now Array handles itself
      // FunctorType(const FunctorType& rv);
      // FunctorType& operator=(const FunctorType& rv);
      virtual void getFunctor(std::unique_ptr<Functor> & r_aptr) =0;
      virtual Functor* getFunctor() =0;
      virtual ~FunctorType();
      std::string getCategory();
      virtual void getInstance(std::unique_ptr<DataItem> & adi, 
			       std::vector<DataItem*> const * args, 
			       LensContext* c);
      virtual void getInstance(std::unique_ptr<DataItem> & adi, 
			       const NDPairList& ndplist,
			       LensContext* c);

   protected:
      DuplicatePointerArray<Functor, 50> _functorList;
   private:
      // not necessary for now Array handles itself
      // void copyContents(const FunctorType& rv);
      // void destructContents();
};
#endif
