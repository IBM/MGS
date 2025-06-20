// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "EdgeDataItem.h"
#include "Edge.h"

// Type
const char* EdgeDataItem::_type = "EDGE";

// Constructors
EdgeDataItem::EdgeDataItem(Edge *edge)
   : _edge(edge)
{
}


EdgeDataItem::EdgeDataItem(const EdgeDataItem& DI)
{
   _edge = DI._edge;
}


// Utility methods
void EdgeDataItem::duplicate(std::unique_ptr<DataItem> & r_aptr) const
{
   r_aptr.reset(new EdgeDataItem(*this));
}


EdgeDataItem& EdgeDataItem::operator=(const EdgeDataItem& DI)
{
   _edge = DI.getEdge();
   return(*this);
}


const char* EdgeDataItem::getType() const
{
   return _type;
}


// Singlet methods

Edge* EdgeDataItem::getEdge() const
{
   return _edge;
}


void EdgeDataItem::setEdge(Edge* e)
{
   _edge = e;
}


EdgeDataItem::~EdgeDataItem()
{
}


std::string EdgeDataItem::getString(Error* error) const
{
   return "";
}
