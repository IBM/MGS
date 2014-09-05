// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
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
void NodeDataItem::duplicate(std::auto_ptr<DataItem> & r_aptr) const
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
