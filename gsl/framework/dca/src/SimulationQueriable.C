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

#include "SimulationQueriable.h"
#ifdef HAVE_MPI
#include <mpi.h>
#endif
#include "PublisherRegistry.h"
#include "PublisherRegistryQueriable.h"
#include "RepertoireQueriable.h"
#include "Simulation.h"
#include "QueryResult.h"
#include "QueryField.h"
#include "EnumEntry.h"
#include "SimulationDataItem.h"
#include "InstanceFactoryRegistry.h"
#include "InstanceFactoryRegistryQueriable.h"

#include <iostream>
#include <string>
#include <sstream>

SimulationQueriable::SimulationQueriable(Simulation* sim)
{
   _sim = sim;
   _publisherQueriable = true;
   _queriableName = _sim->getName();
   _queriableDescription = "Access root repertoire or publisher registry:";
   _queriableType = "Simulation";

   _queriableList.push_back(new RepertoireQueriable(_sim->getRootRepertoire()));
   _queriableList.push_back(new PublisherRegistryQueriable(_sim->getPublisherRegistry()));

   std::unique_ptr<QueryField> aptr_QF(new QueryField(QueryField::ENUM));
   aptr_QF->setName("Simulation Queriables");
   aptr_QF->setDescription("Queriables available from Simulation.");
   aptr_QF->setFormat("");
   std::unique_ptr<EnumEntry> aptrEnumEntry(new EnumEntry("Root Repertoire", "Provide queriable Root Repertoire."));
   aptr_QF->addEnumEntry(aptrEnumEntry);
   aptrEnumEntry.reset(new EnumEntry("Publisher Registry", "Provide queriable Publisher Registry."));
   aptr_QF->addEnumEntry(aptrEnumEntry);

   std::vector<InstanceFactoryRegistry*> const & registries = _sim->getInstanceFactoryRegistries();
   std::vector<InstanceFactoryRegistry*>::const_iterator iter = registries.begin();
   std::vector<InstanceFactoryRegistry*>::const_iterator end = registries.end();
   for (;iter!=end;++iter) {
      InstanceFactoryRegistry* ifr = (*iter);
      std::unique_ptr<InstanceFactoryRegistryQueriable> ifQuer;
      ifr->getQueriable(ifQuer);
      _queriableList.push_back(ifQuer.release());
      std::ostringstream name, description;
      name<<ifr->getTypeName()<<" Registry";
      description<<"Provide queriable "<<name.str();
      aptrEnumEntry.reset(new EnumEntry(name.str(), description.str()));
      aptr_QF->addEnumEntry(aptrEnumEntry);
   }
   _queryDescriptor.addQueryField(aptr_QF);
}


void SimulationQueriable::refresh()
{
   std::vector<InstanceFactoryRegistry*> const & registries = _sim->getInstanceFactoryRegistries();
   std::vector<InstanceFactoryRegistry*>::const_iterator reg_iter = registries.begin();
   std::vector<InstanceFactoryRegistry*>::const_iterator reg_end = registries.end();
   std::list<Queriable*>::iterator qbl_iter = _queriableList.end();
   int j = registries.size();
   for(int i=0; i<j; ++i) {
      --qbl_iter;
      delete (*qbl_iter);
      _queriableList.pop_back();
      qbl_iter = _queriableList.end();
   }
   for (;reg_iter!=reg_end;++reg_iter) {
      InstanceFactoryRegistry* ifr = (*reg_iter);
      std::unique_ptr<InstanceFactoryRegistryQueriable> ifQuer;
      ifr->getQueriable(ifQuer);
      _queriableList.push_back(ifQuer.release());
   }
}


SimulationQueriable::SimulationQueriable(const SimulationQueriable & q)
: Queriable(q), _sim(q._sim)
{
}


void SimulationQueriable::getDataItem(std::unique_ptr<DataItem> & apdi)
{
   SimulationDataItem* di = new SimulationDataItem;
   di->setSimulation(_sim);
   apdi.reset(di);
}


std::unique_ptr<QueryResult> SimulationQueriable::query(int maxItem, int minItem, int searchSize)
{
   std::unique_ptr<QueryResult> qr(new QueryResult());

   // Make sure query field is present
   if (_queryDescriptor.getQueryFields().size()) {
      std::unique_ptr<Queriable> aptr_q;
      std::string field = _queryDescriptor.getQueryFields().front()->getField();
      if (field == "Root Repertoire") {
         aptr_q.reset(new RepertoireQueriable(_sim->getRootRepertoire()));
      }
      else if ((field == "Publisher Registry") || (field == "PublisherRegistry")) {
         aptr_q.reset(new PublisherRegistryQueriable(_sim->getPublisherRegistry()));
      }
      if (aptr_q.get() != 0) {
         qr->addQueriable(aptr_q);
      }
   }
   else std::cerr<<"No query fields found in Simulation!"<<std::endl;
   return qr;
}


Publisher* SimulationQueriable::getQPublisher()
{
   return _sim->getPublisher();
}


void SimulationQueriable::duplicate(std::unique_ptr<Queriable>& dup) const
{
   dup.reset(new SimulationQueriable(*this));
}


SimulationQueriable::~SimulationQueriable()
{
}
