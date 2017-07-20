// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

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
void NodePairDataItem::duplicate(std::auto_ptr<DataItem> & r_aptr) const
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
