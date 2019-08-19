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

#ifndef PUBLISHERREGISTRYQUERIALBE_H
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
