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


void PublisherQueriable::getDataItem(std::auto_ptr<DataItem> & apdi)
{
   PublisherDataItem* di = new PublisherDataItem;
   di->setPublisher(_publisher);
   apdi.reset(di);
}


std::auto_ptr<QueryResult> PublisherQueriable::query(int maxItem, int minItem, int searchSize)
{
   std::auto_ptr<QueryResult> qr(new QueryResult());
   std::cerr<<"Queries not implemented on PublisherQueriable!"<<std::endl;
   return qr;
}


Publisher* PublisherQueriable::getQPublisher()
{
   return _publisher;
}


void PublisherQueriable::duplicate(std::auto_ptr<Queriable>& dup) const
{
   dup.reset(new PublisherQueriable(*this));
}


PublisherQueriable::~PublisherQueriable()
{
}
