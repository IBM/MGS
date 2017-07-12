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
      std::auto_ptr<QueryField> aptr_edgeIdxQF(new QueryField(QueryField::VALUE));
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


void ConnectionSetQueriable::getDataItem(std::auto_ptr<DataItem> & apdi)
{
   ConnectionSetDataItem* di = new ConnectionSetDataItem;
   di->setConnectionSet(_cnxnSet);
   apdi.reset(di);
}


std::auto_ptr<QueryResult> ConnectionSetQueriable::query(int maxItem, int minItem, int searchSize)
{
   std::vector<QueryField*> & queryFields = _queryDescriptor.getQueryFields();
   std::auto_ptr<QueryResult> qr(new QueryResult());
   int idx = -1;
   std::istringstream istr(queryFields[_edgeIdxIdx]->getField());
   istr>>idx;
   if ((idx<(int)_cnxnSet->size()) && (idx>=0)) {
      qr->_numFound = 1;
      std::auto_ptr<Queriable> aptrQueriable(new EdgeQueriable((*_cnxnSet)[idx]));
      qr->addQueriable(aptrQueriable);
   }
   return qr;
}


Publisher* ConnectionSetQueriable::getQPublisher()
{
   return 0;
}


void ConnectionSetQueriable::duplicate(std::auto_ptr<Queriable>& dup) const
{
   dup.reset(new ConnectionSetQueriable(*this));
}


ConnectionSetQueriable::~ConnectionSetQueriable()
{
}
