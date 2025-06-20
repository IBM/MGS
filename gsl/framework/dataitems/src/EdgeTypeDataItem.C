// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "EdgeTypeDataItem.h"
#include "EdgeType.h"

// Type
const char* EdgeTypeDataItem::_type = "EDGE_TYPE";

// Constructors
EdgeTypeDataItem::EdgeTypeDataItem(EdgeType *data)
   : _data(data)
{
}


EdgeTypeDataItem::EdgeTypeDataItem(const EdgeTypeDataItem& DI)
{
   _data = DI._data;
}


// Utility methods
void EdgeTypeDataItem::duplicate(std::unique_ptr<DataItem> & r_aptr) const
{
   r_aptr.reset(new EdgeTypeDataItem(*this));
}


EdgeTypeDataItem& EdgeTypeDataItem::operator=(const EdgeTypeDataItem& DI)
{
   _data = DI.getEdgeType();
   return(*this);
}


const char* EdgeTypeDataItem::getType() const
{
   return _type;
}


// Singlet methods

EdgeType* EdgeTypeDataItem::getEdgeType() const
{
   return _data;
}


void EdgeTypeDataItem::setEdgeType(EdgeType* i)
{
   _data = i;
}
