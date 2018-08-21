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
#include "GeneratedPublisherBase.h"

#include <string>
#include <vector>
#include <iostream>

#include "Service.h"
#include "TriggerType.h"
#include "Simulation.h"
#include "TriggerType.h"

GeneratedPublisherBase::GeneratedPublisherBase(Simulation& sim)
   : _sim(sim)
{
}

GeneratedPublisherBase::GeneratedPublisherBase(
   const GeneratedPublisherBase& rv) : _sim(rv._sim)
{
   copyOwnedHeap(rv);
}

Service* GeneratedPublisherBase::getService(
   const std::string& serviceRequested)
{
   Service* rval = 0;

   rval = internalGetService(serviceRequested);
   if (rval == 0) {
      std::cerr << "Requested service "<< serviceRequested 
		<< " is not found in " << getName() << "'s Service List!"
		<< std::endl;
      exit(-1);
   }
   return rval;
}

Service* GeneratedPublisherBase::getService(
   const std::string& interfaceName, const std::string& subInterfaceName)
{
   Service* rval = 0;
   std::string serviceName = 
      getServiceNameWithInterface(interfaceName, subInterfaceName);
   if (serviceName == "") {
      std::cerr 
	 << "Requested service on interface "<< interfaceName 
	 << " " << subInterfaceName
	 << " is not implemented by " << getName() << "."
	 << std::endl;
      exit(-1);
   }
   rval = internalGetService(serviceName);
   if (rval == 0) {
      std::cerr << "Requested service "<< serviceName 
		<< " is not found in " << getName() << "'s Service List!"
		<< std::endl;
      exit(-1);
   }
   return rval;
}


const std::vector<TriggerType*>& GeneratedPublisherBase::getTriggerDescriptors(
) const
{
   return _triggerDescriptors;
}

TriggerType* GeneratedPublisherBase::getTriggerDescriptor(
   const std::string& triggerDescriptorRequested)
{
   TriggerType* rval=0;
   std::vector<TriggerType*>::iterator it, end = _triggerDescriptors.end();
   for (it = _triggerDescriptors.begin(); it != end; ++it) {
      if ((*it)->getName()==triggerDescriptorRequested) {
	 rval = *it;
      }
   }
   if (rval==0) {
      std::cerr << "Requested trigger descriptor " 
		<< triggerDescriptorRequested << " is not found in " 
		<< getName() << "'s Trigger Descriptor List!"<< std::endl;
      exit(-1);
   }
   return rval;

}

GeneratedPublisherBase::~GeneratedPublisherBase()
{
   destructOwnedHeap();
}

void GeneratedPublisherBase::copyOwnedHeap(const GeneratedPublisherBase& rv)
{
   if (rv._services.size() > 0) {
      std::vector<Service*>::const_iterator it, end = rv._services.end();
      for (it = rv._services.begin(); it != end; ++it) {
	 std::auto_ptr<Service> dup;
	 (*it)->duplicate(dup);
	 _services.push_back(dup.release());
      }
   }
   if (rv._triggerDescriptors.size() > 0) {
      std::vector<TriggerType*>::const_iterator it, 
	 end = rv._triggerDescriptors.end();
      for (it = rv._triggerDescriptors.begin(); it != end; ++it) {
	 std::auto_ptr<TriggerType> dup;
	 (*it)->duplicate(dup);
	 _triggerDescriptors.push_back(dup.release());
      }
   }
}

void GeneratedPublisherBase::destructOwnedHeap()
{
   if (_services.size() > 0) {
      std::vector<Service*>::iterator it, end = _services.end();
      for (it = _services.begin(); it != end; ++it) {
	 delete (*it);
      }
      _services.clear();
   }
   if (_triggerDescriptors.size() > 0) {
      std::vector<TriggerType*>::iterator it, end = _triggerDescriptors.end();
      for (it = _triggerDescriptors.begin(); it != end; ++it) {
	 delete (*it);
      }
      _triggerDescriptors.clear();
   }
}

Service* GeneratedPublisherBase::internalGetService(
   const std::string& serviceRequested)
{
   Service* rval = 0;
   std::vector<Service*>::iterator it, end = _services.end();
   for (it=_services.begin(); it != end; ++it) {
      if ((*it)->getName() == serviceRequested) {
         rval = (*it);
         break;
      }
   }
   if (rval == 0) {
      rval = createService(serviceRequested);
   }
   if (rval == 0) {
      rval = createOptionalService(serviceRequested);
   }
   return rval;
}
