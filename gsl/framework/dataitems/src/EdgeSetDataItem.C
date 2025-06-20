// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "EdgeSetDataItem.h"
#include "EdgeSet.h"
#include "ConnectionSet.h"
#include "Edge.h"

const char* EdgeSetDataItem::_type = "EDGE_SET";

EdgeSetDataItem& EdgeSetDataItem::operator=(const EdgeSetDataItem& DI)
{
   _data = DI.getEdgeSet();
   return(*this);
}


EdgeSetDataItem::EdgeSetDataItem()
   : _data(0)
{
}

EdgeSetDataItem::EdgeSetDataItem(std::unique_ptr<EdgeSet> data)
{
   _data = data.release();
}

EdgeSetDataItem::EdgeSetDataItem(const EdgeSetDataItem& DI)
{
   _data = new EdgeSet(DI._data);
}


void EdgeSetDataItem::duplicate(std::unique_ptr<DataItem> & r_aptr) const
{
   r_aptr.reset(new EdgeSetDataItem(*this));
}


const char* EdgeSetDataItem::getType() const
{
   return _type;
}


// Singlet methods

EdgeSet* EdgeSetDataItem::getEdgeSet(Error* error) const
{
   return _data;
}


void EdgeSetDataItem::setEdgeSet(EdgeSet* i, Error* error)
{
   delete _data;
   _data = new EdgeSet(i);
}


void EdgeSetDataItem::setEdgeSet(ConnectionSet* i, Error* error)
{
   delete _data;
   _data = new EdgeSet();
   _data->addEdges(i);
}


EdgeSetDataItem::~EdgeSetDataItem()
{
   delete _data;
}

std::vector<Triggerable*> EdgeSetDataItem::getTriggerables()
{
   std::vector<Triggerable*> retVal;
   std::vector<Edge*>& edges = _data->getEdges();
   std::vector<Edge*>::iterator it, end = edges.end();
   for(it = edges.begin(); it != end; ++it) {
      retVal.push_back(*it);
   }
   return retVal;
}
