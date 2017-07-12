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
      std::auto_ptr<QueryResult> query(int maxtItem, int minItem, int searchSize);
      Publisher* getQPublisher();
      virtual void duplicate(std::auto_ptr<Queriable>& dup) const;
      void getDataItem(std::auto_ptr<DataItem> &);
      ~GridQueriable();

   private:
      std::vector<int> _size;
      Grid* _grid;
      int _layerIdx;
      int _densityIdxIdx;
};
#endif
