// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "NodeTypeDataItem.h"
#include "NodeType.h"

// Type
const char* NodeTypeDataItem::_type = "NODE_TYPE";

// Constructors
NodeTypeDataItem::NodeTypeDataItem(NodeType *data) 
   : _data(data)
{
}


NodeTypeDataItem::NodeTypeDataItem(const NodeTypeDataItem& DI)
{
   _data = DI._data;
}


// Utility methods
void NodeTypeDataItem::duplicate(std::unique_ptr<DataItem> & r_aptr) const
{
   r_aptr.reset(new NodeTypeDataItem(*this));
}


NodeTypeDataItem& NodeTypeDataItem::operator=(const NodeTypeDataItem& DI)
{
   _data = DI.getNodeType();
   return(*this);
}


const char* NodeTypeDataItem::getType() const
{
   return _type;
}


// Singlet methods

NodeType* NodeTypeDataItem::getNodeType() const
{
   return _data;
}


void NodeTypeDataItem::setNodeType(NodeType* i)
{
   _data = i;
}
