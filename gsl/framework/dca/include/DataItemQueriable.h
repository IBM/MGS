// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef DATAITEMQUERIABLE_H
#define DATAITEMQUERIABLE_H
#include "Copyright.h"

#include "Queriable.h"
#include "QueryResult.h"

#include <string>
#include <memory>


class DataItem;
class QueryField;
class Publisher;

class DataItemQueriable : public Queriable
{
   public:
      DataItemQueriable(std::unique_ptr<DataItem> & dataItem);
      DataItemQueriable(const DataItemQueriable &);
      std::unique_ptr<QueryResult> query(int maxtItem, int minItem, int searchSize);
      Publisher* getQPublisher();
      virtual void duplicate(std::unique_ptr<Queriable>& dup) const;
      void getDataItem(std::unique_ptr<DataItem> &);
      void setName(std::string name);
      void setDescription(std::string description);
      std::string getName() {return _queriableName;}
      std::string getDescription() {return _queriableDescription;}
      ~DataItemQueriable();

   private:
      DataItem* _dataItem;
};
#endif
