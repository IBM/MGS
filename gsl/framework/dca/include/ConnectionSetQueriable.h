// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
