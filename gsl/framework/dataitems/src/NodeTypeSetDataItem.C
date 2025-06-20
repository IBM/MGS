// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "NodeTypeSetDataItem.h"
#include "NodeTypeSet.h"

// Type
const char* NodeTypeSetDataItem::_type = "NODE_TYPE_SET";

// Constructors
NodeTypeSetDataItem::NodeTypeSetDataItem() 
   : _nodeTypeSet(0)
{
}

NodeTypeSetDataItem::NodeTypeSetDataItem(std::unique_ptr<NodeTypeSet> nodeTypeSet) 
{
   _nodeTypeSet = nodeTypeSet.release();
}

NodeTypeSetDataItem::NodeTypeSetDataItem(const NodeTypeSetDataItem& DI)
{
   _nodeTypeSet = DI._nodeTypeSet;
}


// Utility methods
void NodeTypeSetDataItem::duplicate(std::unique_ptr<DataItem> & r_aptr) const
{
   r_aptr.reset(static_cast<DataItem*> (new NodeTypeSetDataItem(*this)));
}


NodeTypeSetDataItem& NodeTypeSetDataItem::operator=(const NodeTypeSetDataItem& DI)
{
   _nodeTypeSet = DI.getNodeTypeSet();
   return(*this);
}


const char* NodeTypeSetDataItem::getType() const
{
   return _type;
}


// Singlet methods

NodeTypeSet* NodeTypeSetDataItem::getNodeTypeSet() const
{
   return _nodeTypeSet;
}


void NodeTypeSetDataItem::setNodeTypeSet(NodeTypeSet* i)
{
   if (i) {
      delete _nodeTypeSet;   
      _nodeTypeSet = new NodeTypeSet(*i);
   }
}


NodeTypeSetDataItem::~NodeTypeSetDataItem()
{
   delete _nodeTypeSet;
}
