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
      void duplicate(std::auto_ptr<DataItem> & r_aptr) const;
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
