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

#ifndef SIMULATIONPUBLISHER_H
#define SIMULATIONPUBLISHER_H
#include "Copyright.h"

#include "Publisher.h"
#include "ServiceDescriptor.h"

#include <string>
#include <vector>
#include <cassert>

class Simulation;
class Service;
class TriggerType;

class SimulationPublisher : public Publisher
{

  public:
  SimulationPublisher(Simulation& Sim);
  SimulationPublisher(const SimulationPublisher& rv);
  virtual const std::vector<ServiceDescriptor>& getServiceDescriptors() const
  {
    return _serviceDescriptors;
  }
  virtual Service* getService(const std::string& serviceRequested);
  virtual Service* getService(const std::string& interfaceName,
                              const std::string& subInterfaceName)
  {
    // implement if needed
    assert(0);
    return 0;
  }
  const std::vector<TriggerType*>& getTriggerDescriptors() const;
  TriggerType* getTriggerDescriptor(
      const std::string& triggerDescriptorRequested);
  std::string getName() const;
  std::string getDescription() const;
  ~SimulationPublisher();
  virtual void duplicate(std::auto_ptr<Publisher>& dup) const;

  private:
  SimulationPublisher& operator=(const SimulationPublisher& rv)
  {
    // disabled;
    assert(0);
    return *this;
  }

  void copyOwnedData(const SimulationPublisher& rv);
  void destructOwnedData();

  Simulation& _sim;
  std::vector<Service*> _services;
  std::vector<TriggerType*> _triggerDescriptors;
  std::string _name;
  std::string _description;
  Publishable* _ptrPublishable;
  static std::vector<ServiceDescriptor> _serviceDescriptors;
};
#endif
