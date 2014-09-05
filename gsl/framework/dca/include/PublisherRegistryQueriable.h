// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
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
      std::auto_ptr<QueryResult> query(int maxtItem, int minItem, int searchSize);
      Publisher* getQPublisher();
      virtual void duplicate(std::auto_ptr<Queriable>& dup) const;
      void getDataItem(std::auto_ptr<DataItem> &);
      ~PublisherRegistryQueriable();

   private:
      PublisherRegistry* _publisherRegistry;
};
#endif
