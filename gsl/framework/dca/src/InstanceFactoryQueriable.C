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

#include "InstanceFactoryQueriable.h"
#include "InstanceFactory.h"
#include "Trigger.h"
#include "InstanceFactory.h"
#include "QueryResult.h"
#include "QueryField.h"
#include "EnumEntry.h"
#include "Simulation.h"
#include "Publisher.h"
#include "InstanceFactoryDataItem.h"
#include "PublisherQueriable.h"
#include "DataItemQueriable.h"

#include <iostream>
#include <sstream>

InstanceFactoryQueriable::InstanceFactoryQueriable(InstanceFactory* instanceFactory)
{
   _instanceFactory = instanceFactory;
   _publisherQueriable = false;
   _queriableName = _instanceFactory->getName();
   _queriableDescription = "Access instances of this type:";
   _queriableType = "Type";

   _instanceQF = new QueryField(QueryField::ENUM);
   _instanceQF->setName("Select");
   _instanceQF->setDescription("Type instances.");
   _instanceQF->setFormat("");

   std::auto_ptr<QueryField> aptr_QF(_instanceQF);
   _queryDescriptor.addQueryField(aptr_QF);
}


void InstanceFactoryQueriable::setName(std::string name)
{
   _queriableName = name;
}


InstanceFactoryQueriable::InstanceFactoryQueriable(const InstanceFactoryQueriable & q)
: Queriable(q), _instanceFactory(q._instanceFactory), _instanceQF(q._instanceQF)
{
}


void InstanceFactoryQueriable::getDataItem(std::auto_ptr<DataItem> & apdi)
{
   InstanceFactoryDataItem* di = new InstanceFactoryDataItem;
   di->setInstanceFactory(_instanceFactory);
   apdi.reset(di);
}


std::auto_ptr<QueryResult> InstanceFactoryQueriable::query(int maxItem, int minItem, int searchSize)
{
   std::auto_ptr<QueryResult> qr(new QueryResult());

   // Make sure query field is present
   if (_queryDescriptor.getQueryFields().size()) {
      std::string field = _queryDescriptor.getQueryFields().front()->getField();
      std::list<Queriable*>::iterator iter = _queriableList.begin();
      std::list<Queriable*>::iterator end = _queriableList.end();
      for (;iter!=end;++iter) {
         if ((*iter)->getQueriableDescriptor().getName() == field) {
            std::auto_ptr<Queriable> aptr_q;
	    (*iter)->duplicate(aptr_q);
            qr->addQueriable(aptr_q);
         }
      }
   }
   else std::cerr<<"No query fields found in InstanceFactory!"<<std::endl;
   return qr;
}


void InstanceFactoryQueriable::duplicate(std::auto_ptr<Queriable>& dup) const
{
   dup.reset(new InstanceFactoryQueriable(*this));
}


void InstanceFactoryQueriable::addQueriable(std::auto_ptr<DataItemQueriable> & q)
{
   std::auto_ptr<EnumEntry> aptrEnumEntry(new EnumEntry(q->getName(), q->getDescription()));
   _instanceQF->addEnumEntry(aptrEnumEntry);
                                 // the queriableList along with its descriptors and their queriables are
   _queriableList.push_back(q.release());
   // deleted in ~Queriable()
}

void InstanceFactoryQueriable::duplicate(
   std::auto_ptr<InstanceFactoryQueriable>& dup) const
{
   dup.reset(new InstanceFactoryQueriable(*this));
}

InstanceFactoryQueriable::~InstanceFactoryQueriable()
{
}
