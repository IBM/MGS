// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef INSTANCEFACTORYREGISTRYQUERIABLE_H
#define INSTANCEFACTORYREGISTRYQUERIABLE_H
#include "Copyright.h"

#include "Queriable.h"

#include <list>
#include <memory>
#include <string>


class InstanceFactoryRegistry;
class QueryField;
class QueryResult;
class QueryDescriptor;
class InstanceFactoryQueriable;

class InstanceFactoryRegistryQueriable : public Queriable
{
   friend class InstanceFactoryRegistry;

   public:
      InstanceFactoryRegistryQueriable(InstanceFactoryRegistry* instanceFactoryRegistry);
      InstanceFactoryRegistryQueriable(const InstanceFactoryRegistryQueriable&);
      std::unique_ptr<QueryResult> query(int maxtItem, int minItem, int searchSize);
      Publisher* getQPublisher() {return 0;}
      virtual void duplicate(std::unique_ptr<Queriable>& dup) const;
      void getDataItem(std::unique_ptr<DataItem> &);
      void addQueriable(std::unique_ptr<InstanceFactoryQueriable> & q);
      ~InstanceFactoryRegistryQueriable();

   private:
      InstanceFactoryRegistry* _instanceFactoryRegistry;
      QueryField* _typeQF;
};
#endif
