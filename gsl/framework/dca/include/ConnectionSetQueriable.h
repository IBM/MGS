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

#ifndef CONNECTIONSETQUERIABLE_H
#define CONNECTIONSETQUERIABLE_H
#include "Copyright.h"

#include "Queriable.h"

#include <list>
#include <string>
#include <vector>
#include <memory>


class ConnectionSet;

class Publisher;
class QueryField;
class QueryResult;
class QueryDescriptor;
class QueriableDescriptor;

class ConnectionSetQueriable : public Queriable
{

   public:
      ConnectionSetQueriable(ConnectionSet* cnxnSet);
      ConnectionSetQueriable(const ConnectionSetQueriable &);
      std::unique_ptr<QueryResult> query(int maxtItem, int minItem, int searchSize);
      Publisher* getQPublisher();
      virtual void duplicate(std::unique_ptr<Queriable>& dup) const;
      void getDataItem(std::unique_ptr<DataItem> &);
      ~ConnectionSetQueriable();

   private:
      ConnectionSet* _cnxnSet;
      int _edgeIdxIdx;
};
#endif
