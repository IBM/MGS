// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
