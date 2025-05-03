// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef CONSTANTTYPE_H
#define CONSTANTTYPE_H
//#include "Copyright.h"

#include "InstanceFactory.h"
#include <memory>
#include <vector>
#include <string>
#include "Constant.h"
#include "DuplicatePointerArray.h"

//class Constant;
class DataItem;
class InstanceFactoryQueriable;
class NDPairList;

class ConstantType : public InstanceFactory
{
   public:
      ConstantType();
      virtual void duplicate(std::unique_ptr<ConstantType>&& dup) const=0;
      virtual void getConstant(std::unique_ptr<Constant> & r_aptr)=0;
      virtual Constant* getConstant()=0;
      virtual std::string getName()=0;
      virtual std::string getDescription()=0;
      virtual ~ConstantType();
      virtual void getInstance(std::unique_ptr<DataItem> & adi, 
			       std::vector<DataItem*> const * args, 
			       GslContext* c);
      virtual void getInstance(std::unique_ptr<DataItem> & adi, 
			       const NDPairList& ndplist,
			       GslContext* c);
   protected:
      DuplicatePointerArray<Constant> _constantList;
};
#endif
