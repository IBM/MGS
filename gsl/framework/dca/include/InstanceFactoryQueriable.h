// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef INSTANCEFACTORYQUERIABLE_H
#define INSTANCEFACTORYQUERIABLE_H
#include "Copyright.h"

#include "Queriable.h"

#include <list>
#include <memory>
#include <string>


class InstanceFactory;
class QueryField;
class QueryResult;
class QueryDescriptor;
class DataItemQueriable;

class InstanceFactoryQueriable : public Queriable
{
   friend class InstanceFactory;

   public:
      InstanceFactoryQueriable(InstanceFactory* instanceFactory);
      InstanceFactoryQueriable(const InstanceFactoryQueriable&);
      std::unique_ptr<QueryResult> query(int maxtItem, int minItem, int searchSize);
      Publisher* getQPublisher() {return 0;}
      void duplicate(std::unique_ptr<InstanceFactoryQueriable>& dup) const;
      virtual void duplicate(std::unique_ptr<Queriable>& dup) const;
      void getDataItem(std::unique_ptr<DataItem> &);
      void setName(std::string name);
      std::string getName() {return _queriableName;}
      std::string getDescription() {return _queriableDescription;}
      void addQueriable(std::unique_ptr<DataItemQueriable> & q);
      ~InstanceFactoryQueriable();

   private:
      InstanceFactory* _instanceFactory;
      QueryField* _instanceQF;
};
#endif
