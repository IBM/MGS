// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef GRIDQUERIABLE_H
#define GRIDQUERIABLE_H
#include "Copyright.h"

#include "Queriable.h"
#include "QueryResult.h"

#include <list>
#include <string>
#include <vector>
#include <memory>


class Grid;
class QueryField;
class Publisher;

class GridQueriable : public Queriable
{
   public:
      GridQueriable(Grid* grid);
      GridQueriable(const GridQueriable &);
      std::unique_ptr<QueryResult> query(int maxtItem, int minItem, int searchSize);
      Publisher* getQPublisher();
      virtual void duplicate(std::unique_ptr<Queriable>& dup) const;
      void getDataItem(std::unique_ptr<DataItem> &);
      ~GridQueriable();

   private:
      std::vector<int> _size;
      Grid* _grid;
      int _layerIdx;
      int _densityIdxIdx;
};
#endif
