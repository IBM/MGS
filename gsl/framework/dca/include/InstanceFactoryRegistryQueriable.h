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
      std::auto_ptr<QueryResult> query(int maxtItem, int minItem, int searchSize);
      Publisher* getQPublisher() {return 0;}
      virtual void duplicate(std::auto_ptr<Queriable>& dup) const;
      void getDataItem(std::auto_ptr<DataItem> &);
      void addQueriable(std::auto_ptr<InstanceFactoryQueriable> & q);
      ~InstanceFactoryRegistryQueriable();

   private:
      InstanceFactoryRegistry* _instanceFactoryRegistry;
      QueryField* _typeQF;
};
#endif
