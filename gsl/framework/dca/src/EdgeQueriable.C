// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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


void EdgeQueriable::getDataItem(std::unique_ptr<DataItem> & apdi)
{
   EdgeDataItem* di = new EdgeDataItem;
   di->setEdge(_edge);
   apdi.reset(di);
}


std::unique_ptr<QueryResult> EdgeQueriable::query(int maxItem, int minItem, int searchSize)
{
   std::unique_ptr<QueryResult> qr(new QueryResult());
   std::cerr<<"Queries not implemented on EdgeQueriable!"<<std::endl;
   return qr;
}


Publisher* EdgeQueriable::getQPublisher()
{
   return _edge->getPublisher();
}


void EdgeQueriable::duplicate(std::unique_ptr<Queriable>& dup) const
{
   dup.reset(new EdgeQueriable(*this));
}


EdgeQueriable::~EdgeQueriable()
{
}
