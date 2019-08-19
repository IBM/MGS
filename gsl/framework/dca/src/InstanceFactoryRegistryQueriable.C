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

#include "InstanceFactoryRegistryQueriable.h"
#include "InstanceFactoryRegistry.h"
#include "InstanceFactoryRegistry.h"
#include "QueryResult.h"
#include "QueryField.h"
#include "EnumEntry.h"
#include "Simulation.h"
#include "Publisher.h"
#include "InstanceFactoryRegistryDataItem.h"
#include "PublisherQueriable.h"
#include "InstanceFactoryQueriable.h"
#include "InstanceFactory.h"

//#include <iostream>
#include <sstream>

InstanceFactoryRegistryQueriable::InstanceFactoryRegistryQueriable(InstanceFactoryRegistry* instanceFactoryRegistry)
{
   _instanceFactoryRegistry = instanceFactoryRegistry;
   _publisherQueriable = false;
   std::ostringstream name, description;
   name<<_instanceFactoryRegistry->getTypeName()<<" Registry";
   _queriableName = name.str();
   description<<"Access types in "<<_queriableName<<".";
   _queriableDescription = description.str();
   _queriableType = "Registry";

   _typeQF = new QueryField(QueryField::ENUM);
   _typeQF->setName("Select");
   _typeQF->setDescription("Types.");
   _typeQF->setFormat("");

   std::list<InstanceFactory*> const & l = _instanceFactoryRegistry->getInstanceFactoryList();
   std::list<InstanceFactory*>::const_iterator iter = l.begin();
   std::list<InstanceFactory*>::const_iterator end = l.end();
   for (; iter != end; ++iter) {
      InstanceFactory* ifc = (*iter);
      std::unique_ptr<InstanceFactoryQueriable> apifq;
      ifc->getQueriable(apifq);
      addQueriable(apifq);
   }
   std::unique_ptr<QueryField> aptr_QF(_typeQF);
   _queryDescriptor.addQueryField(aptr_QF);
}


InstanceFactoryRegistryQueriable::InstanceFactoryRegistryQueriable(const InstanceFactoryRegistryQueriable & q)
: Queriable(q), _instanceFactoryRegistry(q._instanceFactoryRegistry), _typeQF(q._typeQF)
{
}


void InstanceFactoryRegistryQueriable::getDataItem(std::unique_ptr<DataItem> & apdi)
{
   InstanceFactoryRegistryDataItem* di = new InstanceFactoryRegistryDataItem;
   di->setInstanceFactoryRegistry(_instanceFactoryRegistry);
   apdi.reset(di);
}


std::unique_ptr<QueryResult> InstanceFactoryRegistryQueriable::query(int maxItem, int minItem, int searchSize)
{
   std::unique_ptr<QueryResult> qr(new QueryResult());

   // Make sure query field is present
   if (_queryDescriptor.getQueryFields().size()) {
      std::string field = _queryDescriptor.getQueryFields().front()->getField();
      std::list<Queriable*>::iterator iter = _queriableList.begin();
      std::list<Queriable*>::iterator end = _queriableList.end();
      for (;iter!=end;++iter) {
         if ((*iter)->getQueriableDescriptor().getName() == field) {
            std::unique_ptr<Queriable> aptr_q;
	    (*iter)->duplicate(aptr_q);
            qr->addQueriable(aptr_q);
         }
      }
   }
   else std::cerr<<"No query fields found in InstanceFactoryRegistry!"<<std::endl;
   return qr;
}


void InstanceFactoryRegistryQueriable::duplicate(std::unique_ptr<Queriable>& dup) const
{
   dup.reset(new InstanceFactoryRegistryQueriable(*this));
}


void InstanceFactoryRegistryQueriable::addQueriable(std::unique_ptr<InstanceFactoryQueriable> & q)
{
   std::unique_ptr<EnumEntry> aptrEnumEntry(new EnumEntry(q->getName(), q->getDescription()));
   _typeQF->addEnumEntry(aptrEnumEntry);
                                 // the queriableList along with its descriptors and their queriables are
   _queriableList.push_back(q.release());
   // deleted in ~Queriable()
}


InstanceFactoryRegistryQueriable::~InstanceFactoryRegistryQueriable()
{
}
