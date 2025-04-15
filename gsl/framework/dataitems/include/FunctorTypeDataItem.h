// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef FUNCTORTYPEDATAITEM_H
#define FUNCTORTYPEDATAITEM_H
#include "Copyright.h"

#include "InstanceFactoryDataItem.h"
#include "DataItem.h"
#include <string>
#include <list>
class FunctorType;
class C_parameter_type;
class InstanceFactory;

class FunctorTypeDataItem : public InstanceFactoryDataItem
{
   public:
      static const char *_type;
   private:
      FunctorType* _functorType;
      std::list<C_parameter_type> *constructor_ptl;
      std::list<C_parameter_type> *function_ptl;
      std::list<C_parameter_type> *return_ptl;
      std::string category;

   public:
      FunctorTypeDataItem();
      FunctorTypeDataItem(FunctorTypeDataItem const *);
      ~FunctorTypeDataItem();

      // DataItem Methods go here
      void duplicate(std::unique_ptr<DataItem> & r_aptr) const;
      const char* getType() const;

      // FunctorType accessors
      virtual void setFunctorType(FunctorType* ft);
      FunctorType* getFunctorType() const;
      void setCategory(std::string cat);
      std::string getCategory() const;
      void setConstructorParams(std::list<C_parameter_type> *ptl);
      std::list<C_parameter_type> * getConstructorParams() const;
      void setFunctionParams(std::list<C_parameter_type> *ptl);
      std::list<C_parameter_type> * getFunctionParams() const;
      void setReturnParams(std::list<C_parameter_type> *ptl);
      std::list<C_parameter_type> * getReturnParams() const;
      InstanceFactory* getInstanceFactory() const;
      void setInstanceFactory(InstanceFactory*);

   protected:
      DataItem &assign(const DataItem &);
};
#endif
