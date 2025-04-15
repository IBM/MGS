// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef STRUCTTYPE_H
#define STRUCTTYPE_H
#include "Copyright.h"

#include "InstanceFactory.h"
#include <memory>
#include <vector>
#include <list>
#include <string>

class Struct;
class DataItem;
class InstanceFactoryQueriable;
class NDPairList;

class StructType : public InstanceFactory
{
   public:
      StructType();
      StructType(const StructType& rv);
      StructType& operator=(const StructType& rv);
      virtual void duplicate(std::unique_ptr<StructType>&& dup) const=0;
      virtual void getStruct(std::unique_ptr<Struct>& r_aptr)=0;
      virtual Struct* getStruct()=0;
      virtual std::string getName()=0;
      virtual std::string getDescription()=0;
      virtual ~StructType();
      virtual void getInstance(std::unique_ptr<DataItem>& adi, 
			       std::vector<DataItem*> const * args, 
			       LensContext* c);
      virtual void getInstance(std::unique_ptr<DataItem>& adi, 
			       const NDPairList& ndplist,
			       LensContext* c);
   protected:
      std::list<Struct*> _structList;      
   private:
      void copyContents(const StructType& rv);
      void destructContents();
};
#endif
