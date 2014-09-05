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

#ifndef GeneratedPublisherBase_H
#define GeneratedPublisherBase_H
#include "Copyright.h"

#include <string>
#include <vector>

#include "Publisher.h"

class Service;
class TriggerType;
class Simulation;
class TriggerType;

class GeneratedPublisherBase : public Publisher
{

   public:
      GeneratedPublisherBase(Simulation& sim);
      GeneratedPublisherBase(const GeneratedPublisherBase& rv);
      virtual Service* getService(const std::string& serviceRequested);
      virtual Service* getService(const std::string& interfaceName,
				  const std::string& subInterfaceName);
      virtual const std::vector<TriggerType*>& getTriggerDescriptors() const;
      virtual TriggerType* getTriggerDescriptor(
	 const std::string& triggerDescriptorRequested);
      virtual ~GeneratedPublisherBase();

   private:
      void copyOwnedHeap(const GeneratedPublisherBase& rv);
      void destructOwnedHeap();

      // Disabled, there is a reference member, true op= is not possible,
      // new object has to be created.
      GeneratedPublisherBase& operator=(const GeneratedPublisherBase& rv);

      inline Service* internalGetService(const std::string& serviceRequested);

   protected:
      Simulation& _sim;
      std::vector<Service*> _services;
      std::vector<TriggerType*> _triggerDescriptors;

      virtual Service* createService(
	 const std::string& serviceRequested) = 0;
      virtual Service* createOptionalService(
	 const std::string& serviceRequested) = 0;
      virtual std::string getServiceNameWithInterface(
	 const std::string& interfaceName,
	 const std::string& subInterfaceName) = 0;
};
#endif
