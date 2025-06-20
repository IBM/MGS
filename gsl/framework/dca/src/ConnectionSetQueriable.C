// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "ConnectionSetQueriable.h"
#include "GridLayerDescriptor.h"
#include "ConnectionSet.h"
#include "QueryField.h"
#include "QueryResult.h"
#include "QueriableDescriptor.h"
#include "EdgeQueriable.h"
#include "ConnectionSetDataItem.h"

#include <sstream>

ConnectionSetQueriable::ConnectionSetQueriable(ConnectionSet* cnxnSet)
{

   _edgeIdxIdx = -1;             // initialize to -1 for sake of safe copy construction only

   _cnxnSet = cnxnSet;
   _publisherQueriable = false;
   _queriableName = _cnxnSet->getName();
   _queriableDescription = "Access edges in a connection set by index:";
   _queriableType = "Connection Set";
   if (_cnxnSet->size() > 0) {
      std::unique_ptr<QueryField> aptr_edgeIdxQF(new QueryField(QueryField::VALUE));
      aptr_edgeIdxQF->setName("Edge Index");
      aptr_edgeIdxQF->setDescription("Index of edge.");
      std::ostringstream ostr;
      ostr<<"[0.."<<_cnxnSet->size()-1<<"]";
      aptr_edgeIdxQF->setFormat(ostr.str());
      _edgeIdxIdx = _queryDescriptor.addQueryField(aptr_edgeIdxQF);
   }
}


ConnectionSetQueriable::ConnectionSetQueriable(const ConnectionSetQueriable & q)
: Queriable(q), _cnxnSet(q._cnxnSet), _edgeIdxIdx(q._edgeIdxIdx)
{
}


void ConnectionSetQueriable::getDataItem(std::unique_ptr<DataItem> & apdi)
{
   ConnectionSetDataItem* di = new ConnectionSetDataItem;
   di->setConnectionSet(_cnxnSet);
   apdi.reset(di);
}


std::unique_ptr<QueryResult> ConnectionSetQueriable::query(int maxItem, int minItem, int searchSize)
{
   std::vector<QueryField*> & queryFields = _queryDescriptor.getQueryFields();
   std::unique_ptr<QueryResult> qr(new QueryResult());
   int idx = -1;
   std::istringstream istr(queryFields[_edgeIdxIdx]->getField());
   istr>>idx;
   if ((idx<(int)_cnxnSet->size()) && (idx>=0)) {
      qr->_numFound = 1;
      std::unique_ptr<Queriable> aptrQueriable(new EdgeQueriable((*_cnxnSet)[idx]));
      qr->addQueriable(aptrQueriable);
   }
   return qr;
}


Publisher* ConnectionSetQueriable::getQPublisher()
{
   return 0;
}


void ConnectionSetQueriable::duplicate(std::unique_ptr<Queriable>& dup) const
{
   dup.reset(new ConnectionSetQueriable(*this));
}


ConnectionSetQueriable::~ConnectionSetQueriable()
{
}
