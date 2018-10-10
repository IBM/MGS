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

#ifndef EDGEQUERIALBE_H
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
