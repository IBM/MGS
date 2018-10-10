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

#ifdef HAVE_MPI
#include <mpi.h>
#endif
#include "PublisherQueriable.h"
#include "QueryResult.h"
#include "QueryField.h"
#include "EnumEntry.h"
#include "PublisherDataItem.h"
#include "Publisher.h"

#include <iostream>
#include <sstream>

PublisherQueriable::PublisherQueriable(Publisher* entryPublisher)
{
   _publisher = entryPublisher;
   _publisherQueriable = true;
   _queriableName = _publisher->getName();
   _queriableDescription = _publisher->getDescription();
   _queriableType = "Registry Entry";
}


PublisherQueriable::PublisherQueriable(const PublisherQueriable& q)
   : Queriable(q), _publisher(q._publisher)
{
}


void PublisherQueriable::getDataItem(std::unique_ptr<DataItem> & apdi)
{
   PublisherDataItem* di = new PublisherDataItem;
   di->setPublisher(_publisher);
   apdi.reset(di);
}


std::unique_ptr<QueryResult> PublisherQueriable::query(int maxItem, int minItem, int searchSize)
{
   std::unique_ptr<QueryResult> qr(new QueryResult());
   std::cerr<<"Queries not implemented on PublisherQueriable!"<<std::endl;
   return qr;
}


Publisher* PublisherQueriable::getQPublisher()
{
   return _publisher;
}


void PublisherQueriable::duplicate(std::unique_ptr<Queriable>& dup) const
{
   dup.reset(new PublisherQueriable(*this));
}


PublisherQueriable::~PublisherQueriable()
{
}
