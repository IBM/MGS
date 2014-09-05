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

#include "NodeTypeSetDataItem.h"
#include "NodeTypeSet.h"

// Type
const char* NodeTypeSetDataItem::_type = "NODE_TYPE_SET";

// Constructors
NodeTypeSetDataItem::NodeTypeSetDataItem() 
   : _nodeTypeSet(0)
{
}

NodeTypeSetDataItem::NodeTypeSetDataItem(std::auto_ptr<NodeTypeSet> nodeTypeSet) 
{
   _nodeTypeSet = nodeTypeSet.release();
}

NodeTypeSetDataItem::NodeTypeSetDataItem(const NodeTypeSetDataItem& DI)
{
   _nodeTypeSet = DI._nodeTypeSet;
}


// Utility methods
void NodeTypeSetDataItem::duplicate(std::auto_ptr<DataItem> & r_aptr) const
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
