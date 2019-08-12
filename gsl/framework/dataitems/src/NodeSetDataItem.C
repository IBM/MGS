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

#include "NodeSetDataItem.h"
#include "NodeSet.h"
#include "Node.h"
#include <sstream>

// Type
const char* NodeSetDataItem::_type = "NODE_SET";

// Constructors
NodeSetDataItem::NodeSetDataItem()
   : _nodeset(0)
{
}

NodeSetDataItem::NodeSetDataItem(std::unique_ptr<NodeSet>& nodeset)
{
   _nodeset = nodeset.release();
}

NodeSetDataItem::NodeSetDataItem(const NodeSetDataItem& DI)
   : _nodeset(0)
{
   if (DI._nodeset) _nodeset = new NodeSet(*DI._nodeset);
}

// Utility methods
void NodeSetDataItem::duplicate(std::unique_ptr<DataItem> & r_aptr) const
{
   r_aptr.reset(static_cast<DataItem*> (new NodeSetDataItem(*this)));
}

NodeSetDataItem& NodeSetDataItem::operator=(const NodeSetDataItem& DI)
{
   delete _nodeset;
   if (DI.getNodeSet())_nodeset = new NodeSet(*DI.getNodeSet());
   else _nodeset = 0;
   return(*this);
}

const char* NodeSetDataItem::getType() const
{
   return _type;
}

// Singlet methods

NodeSet* NodeSetDataItem::getNodeSet() const
{
   return _nodeset;
}

void NodeSetDataItem::setNodeSet(GridSet* ns)
{
   delete _nodeset;
   if (ns) _nodeset = new NodeSet(*ns);
   else _nodeset = 0;
}

void NodeSetDataItem::setNodeSet(NodeSet* ns)
{
   delete _nodeset;
   if (ns) _nodeset = new NodeSet(*ns);
   else _nodeset = 0;
}

NodeSetDataItem::~NodeSetDataItem()
{
   delete _nodeset;
}

std::string NodeSetDataItem::getString(Error* error) const
{
   std::ostringstream ostr;
   ostr << *((GridSet*)_nodeset);
   return ostr.str();
}

std::vector<Triggerable*> NodeSetDataItem::getTriggerables()
{
   std::vector<NodeDescriptor*> nodes;
   std::vector<Triggerable*> retVal;
   _nodeset->getNodes(nodes);
   std::vector<NodeDescriptor*>::iterator it, end = nodes.end();
   for(it = nodes.begin(); it != end; ++it) {
      // @TODO Distributed local filter
      if ((*it)->getNode())       // added by Jizhu Lu on 12/04/2005
         retVal.push_back((*it)->getNode());
   }
   return retVal;
}
