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

#include "EdgeQueriable.h"
#include "GridLayerDescriptor.h"
#include "Edge.h"
#include "QueryResult.h"
#include "QueryField.h"
#include "EnumEntry.h"
#include "EdgeDataItem.h"

//#include <iostream>
#include <sstream>

EdgeQueriable::EdgeQueriable(Edge* edge)
{
   _edge = edge;
   _publisherQueriable = true;
   std::ostringstream name;
   name<<_edge->getModelName()<<" Edge";
   _queriableName = name.str();
   _queriableDescription = "Access the edge's publisher:";
   _queriableType = "Edge";
}


EdgeQueriable::EdgeQueriable(const EdgeQueriable & q)
   : Queriable(q), _edge(q._edge)
{
}


void EdgeQueriable::getDataItem(std::auto_ptr<DataItem> & apdi)
{
   EdgeDataItem* di = new EdgeDataItem;
   di->setEdge(_edge);
   apdi.reset(di);
}


std::auto_ptr<QueryResult> EdgeQueriable::query(int maxItem, int minItem, int searchSize)
{
   std::auto_ptr<QueryResult> qr(new QueryResult());
   std::cerr<<"Queries not implemented on EdgeQueriable!"<<std::endl;
   return qr;
}


Publisher* EdgeQueriable::getQPublisher()
{
   return _edge->getPublisher();
}


void EdgeQueriable::duplicate(std::auto_ptr<Queriable>& dup) const
{
   dup.reset(new EdgeQueriable(*this));
}


EdgeQueriable::~EdgeQueriable()
{
}
