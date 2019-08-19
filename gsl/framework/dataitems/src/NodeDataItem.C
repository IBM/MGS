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

#include "NodeDataItem.h"
#include "Node.h"

// Type
const char* NodeDataItem::_type = "NODE";

// Constructors
NodeDataItem::NodeDataItem(Node *node) 
   : _node(node)
{
}


NodeDataItem::NodeDataItem(const NodeDataItem& DI)
{
   _node = DI._node;
}


// Utility methods
void NodeDataItem::duplicate(std::unique_ptr<DataItem> & r_aptr) const
{
   r_aptr.reset(new NodeDataItem(*this));
}


NodeDataItem& NodeDataItem::operator=(const NodeDataItem& DI)
{
   _node = DI.getNode();
   return(*this);
}


const char* NodeDataItem::getType() const
{
   return _type;
}


// Singlet methods

Node* NodeDataItem::getNode() const
{
   return _node;
}


void NodeDataItem::setNode(Node* ns)
{
   _node = ns;
}


NodeDataItem::~NodeDataItem()
{
}


std::string NodeDataItem::getString(Error* error) const
{
   return "";
}
