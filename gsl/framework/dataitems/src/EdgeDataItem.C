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
void EdgeDataItem::duplicate(std::auto_ptr<DataItem> & r_aptr) const
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
