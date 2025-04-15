// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "NodePairDataItem.h"
#include "Node.h"

// Type
const char* NodePairDataItem::_type = "NODE";

// Constructors
NodePairDataItem::NodePairDataItem() : _first(0), _second(0)
{
}


NodePairDataItem::NodePairDataItem(const NodePairDataItem& DI)
{
   _first = DI._first;
   _second = DI._second;
}


// Utility methods
void NodePairDataItem::duplicate(std::unique_ptr<DataItem> & r_aptr) const
{
   r_aptr.reset(static_cast<DataItem*> (new NodePairDataItem(*this)));
}


NodePairDataItem& NodePairDataItem::operator=(const NodePairDataItem& DI)
{
   _first = DI.getFirstNode();
   _second = DI.getSecondNode();
   return(*this);
}


const char* NodePairDataItem::getType() const
{
   return _type;
}


// Singlet methods

Node* NodePairDataItem::getFirstNode() const
{
   return _first;
}


void NodePairDataItem::setFirstNode(Node* ns)
{
   _first = ns;
}


Node* NodePairDataItem::getSecondNode() const
{
   return _second;
}


void NodePairDataItem::setSecondNode(Node* ns)
{
   _second = ns;
}


NodePairDataItem::~NodePairDataItem()
{
}


std::string NodePairDataItem::getString(Error* error) const
{
   return "";
}
