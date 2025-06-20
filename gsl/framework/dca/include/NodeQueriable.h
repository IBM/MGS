// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef NODEQUERIABLE_H
#define NODEQUERIABLE_H
#include "Copyright.h"

#include "Queriable.h"

#include <list>
#include <memory>


class NodeDescriptor;
class QueryField;
class QueryResult;
class QueryDescriptor;

class NodeQueriable : public Queriable
{
   public:
      NodeQueriable(NodeDescriptor* nodeDescriptor);
      NodeQueriable(const NodeQueriable&);
      std::unique_ptr<QueryResult> query(int maxtItem, int minItem, int searchSize);
      Publisher* getQPublisher();
      virtual void duplicate(std::unique_ptr<Queriable>& dup) const;
      void getDataItem(std::unique_ptr<DataItem> &);
      ~NodeQueriable();

   private:
      NodeDescriptor* _nodeDescriptor;
};
#endif
