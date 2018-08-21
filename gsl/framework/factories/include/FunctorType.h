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
      virtual void getFunctor(std::auto_ptr<Functor> & r_aptr) =0;
      virtual Functor* getFunctor() =0;
      virtual ~FunctorType();
      std::string getCategory();
      virtual void getInstance(std::auto_ptr<DataItem> & adi, 
			       std::vector<DataItem*> const * args, 
			       LensContext* c);
      virtual void getInstance(std::auto_ptr<DataItem> & adi, 
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
