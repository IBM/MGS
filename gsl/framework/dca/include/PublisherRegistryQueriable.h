// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef PUBLISHERREGISTRYQUERIABLE_H
#define PUBLISHERREGISTRYQUERIABLE_H
#include "Copyright.h"

#include "Queriable.h"

#include <list>
#include <memory>
#include <string>


class PublisherRegistry;
class QueryField;
class QueryResult;
class QueryDescriptor;

class PublisherRegistryQueriable : public Queriable
{

   public:
      PublisherRegistryQueriable(PublisherRegistry* publisherRegistry);
      PublisherRegistryQueriable(const PublisherRegistryQueriable&);
      std::unique_ptr<QueryResult> query(int maxtItem, int minItem, int searchSize);
      Publisher* getQPublisher();
      virtual void duplicate(std::unique_ptr<Queriable>& dup) const;
      void getDataItem(std::unique_ptr<DataItem> &);
      ~PublisherRegistryQueriable();

   private:
      PublisherRegistry* _publisherRegistry;
};
#endif
