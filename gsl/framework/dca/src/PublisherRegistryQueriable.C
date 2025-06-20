// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifdef HAVE_MPI
#include <mpi.h>
#endif
#include "PublisherRegistryQueriable.h"
#include "PublisherRegistry.h"
#include "PublisherRegistry.h"
#include "QueryResult.h"
#include "QueryField.h"
#include "EnumEntry.h"
#include "Simulation.h"
#include "Publisher.h"
#include "PublisherRegistryDataItem.h"
#include "PublisherQueriable.h"

#include <iostream>
#include <sstream>

PublisherRegistryQueriable::PublisherRegistryQueriable(PublisherRegistry* publisherRegistry)
{
   _publisherRegistry = publisherRegistry;
   _publisherQueriable = false;
   _queriableName = "Publisher Registry";
   _queriableDescription = "Access entries in the Publisher Registry:";
   _queriableType = "Publisher Registry";

   std::unique_ptr<QueryField> aptr_QF(new QueryField(QueryField::ENUM));
   aptr_QF->setName("Publisher Registry Queriables");
   aptr_QF->setDescription("Publisher Registry's queriables entries.");
   aptr_QF->setFormat("");

   const std::list<Publisher*>& l = _publisherRegistry->getPublisherList();
   std::list<Publisher*>::const_iterator iter = l.begin();
   std::list<Publisher*>::const_iterator end = l.end();
   for (; iter != end; ++iter) {
      Publisher* pub = (*iter);
      std::unique_ptr<EnumEntry> aptrEnumEntry(new EnumEntry(pub->getName(), pub->getDescription()));
      aptr_QF->addEnumEntry(aptrEnumEntry);
      _queriableList.push_back(new PublisherQueriable(*iter));
   }
   _queryDescriptor.addQueryField(aptr_QF);
}


PublisherRegistryQueriable::PublisherRegistryQueriable(
   const PublisherRegistryQueriable & q)
: Queriable(q), _publisherRegistry(q._publisherRegistry)
{
}


void PublisherRegistryQueriable::getDataItem(std::unique_ptr<DataItem> & apdi)
{
   PublisherRegistryDataItem* di = new PublisherRegistryDataItem;
   di->setPublisherRegistry(_publisherRegistry);
   apdi.reset(di);
}


std::unique_ptr<QueryResult> PublisherRegistryQueriable::query(int maxItem, int minItem, int searchSize)
{
   std::unique_ptr<QueryResult> qr(new QueryResult());

   // Make sure query field is present
   if (_queryDescriptor.getQueryFields().size()) {
      std::string field = _queryDescriptor.getQueryFields().front()->getField();
      std::unique_ptr<Queriable> aptr_q(new PublisherQueriable(_publisherRegistry->getPublisher(field)));
      qr->addQueriable(aptr_q);
   }
   else std::cerr<<"No query fields found in Publisher Registry!"<<std::endl;
   return qr;
}


Publisher* PublisherRegistryQueriable::getQPublisher()
{
   return 0;
}

void PublisherRegistryQueriable::duplicate(std::unique_ptr<Queriable>& dup) const
{
   dup.reset(new PublisherRegistryQueriable(*this));
}


PublisherRegistryQueriable::~PublisherRegistryQueriable()
{
}
