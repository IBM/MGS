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

EdgeSetDataItem::EdgeSetDataItem(std::auto_ptr<EdgeSet> data)
{
   _data = data.release();
}

EdgeSetDataItem::EdgeSetDataItem(const EdgeSetDataItem& DI)
{
   _data = new EdgeSet(DI._data);
}


void EdgeSetDataItem::duplicate(std::auto_ptr<DataItem> & r_aptr) const
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
