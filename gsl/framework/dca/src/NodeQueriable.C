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

#ifdef HAVE_MPI
#include <mpi.h>
#endif
#include "NodeQueriable.h"
#include "GridLayerDescriptor.h"
#include "Node.h"
#include "QueryResult.h"
#include "QueryField.h"
#include "EnumEntry.h"
#include "NodeDataItem.h"

#include <iostream>
#include <sstream>

NodeQueriable::NodeQueriable(NodeDescriptor* nodeDescriptor)
{
   _nodeDescriptor = nodeDescriptor;
   _publisherQueriable = true;
   std::ostringstream name;
   name<<_nodeDescriptor->getGridLayerDescriptor()->getModelName()<<" Node";
   _queriableName = name.str();
   _queriableDescription = "Access the node's publisher:";
   _queriableType = "Node";
}


NodeQueriable::NodeQueriable(const NodeQueriable & q)
   : Queriable(q), _nodeDescriptor(q._nodeDescriptor)
{
}


void NodeQueriable::getDataItem(std::auto_ptr<DataItem> & apdi)
{
   NodeDataItem* di = new NodeDataItem;
   // WARNING: THIS NODE CAN AND WILL OFTEN BE NULL IN A DISTRIBUTED RUN. ARCHITECTURE NEEDS WORK HERE.
   if (_nodeDescriptor->getNode()==0) std::cerr<<"WARNING: NULL pointer on NodeQueriable!"<<std::endl;
   di->setNode(_nodeDescriptor->getNode());
   apdi.reset(di);
}


std::auto_ptr<QueryResult> NodeQueriable::query(int maxItem, int minItem, int searchSize)
{
   std::auto_ptr<QueryResult> qr(new QueryResult());
   std::cerr<<"Queries not implemented on NodeQueriable!"<<std::endl;
   return qr;
}


Publisher* NodeQueriable::getQPublisher()
{
   return _nodeDescriptor->getPublisher();
}


void NodeQueriable::duplicate(std::auto_ptr<Queriable>& dup) const
{
   dup.reset(new NodeQueriable(*this));
}


NodeQueriable::~NodeQueriable()
{
}
