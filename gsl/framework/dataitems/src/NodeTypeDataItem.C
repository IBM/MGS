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
void NodeTypeDataItem::duplicate(std::auto_ptr<DataItem> & r_aptr) const
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
