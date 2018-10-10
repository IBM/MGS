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
