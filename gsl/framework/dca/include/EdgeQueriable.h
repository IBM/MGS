// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef EDGEQUERIABLE_H
#define EDGEQUERIABLE_H
#include "Copyright.h"

#include "Queriable.h"

#include <list>
#include <memory>


class Edge;
class QueryField;
class QueryResult;
class QueryDescriptor;

class EdgeQueriable : public Queriable
{

   public:
      EdgeQueriable(Edge* edge);
      EdgeQueriable(const EdgeQueriable&);
      std::unique_ptr<QueryResult> query(int maxtItem, int minItem, int searchSize);
      Publisher* getQPublisher();
      virtual void duplicate(std::unique_ptr<Queriable>& dup) const;
      void getDataItem(std::unique_ptr<DataItem> &);
      ~EdgeQueriable();

   private:
      Edge* _edge;
};
#endif
