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

#ifdef HAVE_MPI
#include <mpi.h>
#endif
#include "Simulation.h"
#include "SimulationPublisher.h"
#include "GenericService.h"
// #include "UnsignedService.h"
// #include "NullService.h"
#include "TriggerType.h"

#include <iostream>

SimulationPublisher::SimulationPublisher(Simulation& Sim)
    : _sim(Sim), _ptrPublishable(&_sim)
{
  {
		//add 1
    Service* s =
        new GenericService<unsigned>(_ptrPublishable, &(_sim._iteration));
    _services.push_back(s);
  }

  //{
	//	//add 2
  //  Service* s =
  //      new GenericService<float>(_ptrPublishable, &(_sim._iteration));
  //  _services.push_back(s);
  //}
  //    s = new NullService(_sim);
  //    _services.push_back(s);

  _name = "SimulationPublisher";
  _description = "Publishes data from Simulation.";

  if (_serviceDescriptors.size() == 0)
  {
    _serviceDescriptors.push_back(ServiceDescriptor(
        "Iteration", "Returns iteration number of simulation.", "unsigned"));
  //  _serviceDescriptors.push_back(ServiceDescriptor(
  //      "CurrentTime", "Returns the current simulation time.", "float"));
  }
}

SimulationPublisher::SimulationPublisher(const SimulationPublisher& rv)
    : _sim(rv._sim),
      _name(rv._name),
      _description(rv._description),
      _ptrPublishable(rv._ptrPublishable)
{
  copyOwnedData(rv);
}

Service* SimulationPublisher::getService(const std::string& serviceRequested)
{
  Service* rval = 0;
  std::vector<Service*>::iterator end = _services.end();
  for (std::vector<Service*>::iterator iter = _services.begin(); iter != end;
       ++iter)
  {
    if ((*iter)->getName() == serviceRequested)
    {
      rval = (*iter);
      break;
    }
  }
  if (rval == 0)
  {
    std::cerr << "Requested service " << serviceRequested
              << " not found in SimulationPublisher's Service List!"
              << std::endl;
    exit(-1);
  }
  return rval;  // Throw exception here.
}

const std::vector<TriggerType*>& SimulationPublisher::getTriggerDescriptors()
    const
{
  return _triggerDescriptors;
}

TriggerType* SimulationPublisher::getTriggerDescriptor(
    const std::string& triggerDescriptorRequested)
{
  std::vector<TriggerType*>::iterator end = _triggerDescriptors.end();
  for (std::vector<TriggerType*>::iterator iter = _triggerDescriptors.begin();
       iter != end; iter++)
  {
    if ((*iter)->getName() == triggerDescriptorRequested) return *iter;
  }
  std::cerr
      << "Requested trigger descriptor not found in Trigger Descriptor List!"
      << std::endl;
  return 0;  // Throw exception here.
}

std::string SimulationPublisher::getName() const { return _name; }

std::string SimulationPublisher::getDescription() const { return _description; }

SimulationPublisher::~SimulationPublisher() { destructOwnedData(); }

void SimulationPublisher::copyOwnedData(const SimulationPublisher& rv)
{
  if (rv._services.size() > 0)
  {
    std::vector<Service*>::const_iterator it, end = rv._services.end();
    std::auto_ptr<Service> dup;
    for (it = rv._services.begin(); it != end; ++it)
    {
      (*it)->duplicate(dup);
      _services.push_back(dup.release());
    }
  }
  if (rv._triggerDescriptors.size() > 0)
  {
    std::vector<TriggerType*>::const_iterator it,
        end = rv._triggerDescriptors.end();
    std::auto_ptr<TriggerType> dup;
    for (it = rv._triggerDescriptors.begin(); it != end; ++it)
    {
      (*it)->duplicate(dup);
      _triggerDescriptors.push_back(dup.release());
    }
  }
}

void SimulationPublisher::destructOwnedData()
{
  if (_services.size() > 0)
  {
    std::vector<Service*>::iterator it, end = _services.end();
    for (it = _services.begin(); it != end; ++it)
    {
      delete (*it);
    }
    _services.clear();
  }
  if (_triggerDescriptors.size() > 0)
  {
    std::vector<TriggerType*>::iterator it, end = _triggerDescriptors.end();
    for (it = _triggerDescriptors.begin(); it != end; ++it)
    {
      delete (*it);
    }
    _triggerDescriptors.clear();
  }
}

void SimulationPublisher::duplicate(std::auto_ptr<Publisher>& dup) const
{
  dup.reset(new SimulationPublisher(*this));
}

std::vector<ServiceDescriptor> SimulationPublisher::_serviceDescriptors;
