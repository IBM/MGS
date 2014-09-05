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
      DataItemQueriable(std::auto_ptr<DataItem> & dataItem);
      DataItemQueriable(const DataItemQueriable &);
      std::auto_ptr<QueryResult> query(int maxtItem, int minItem, int searchSize);
      Publisher* getQPublisher();
      virtual void duplicate(std::auto_ptr<Queriable>& dup) const;
      void getDataItem(std::auto_ptr<DataItem> &);
      void setName(std::string name);
      void setDescription(std::string description);
      std::string getName() {return _queriableName;}
      std::string getDescription() {return _queriableDescription;}
      ~DataItemQueriable();

   private:
      DataItem* _dataItem;
};
#endif
