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

#ifndef PUBLISHER_H
#define PUBLISHER_H
#include "Copyright.h"

#include <memory>
#include <string>
#include <vector>
#include <cassert>
#include "Publishable.h"

class ServiceDescriptor;
class Service;
class TriggerType;

class Publisher
{

   public:
      virtual Service* getService(const std::string& serviceRequested) =0;
      virtual Service* getService(const std::string& interfaceName,
				  const std::string& subInterfaceName) =0;
      virtual const std::vector<TriggerType*>& 
      getTriggerDescriptors() const =0;
      virtual TriggerType* getTriggerDescriptor(
	 const std::string& triggerDescriptorRequested) =0;
      virtual std::string getName() const =0;
      virtual std::string getDescription() const =0;
      virtual const std::vector<ServiceDescriptor>& 
      getServiceDescriptors() const = 0;
      virtual ~Publisher() {}

      virtual void duplicate(std::unique_ptr<Publisher>& dup) const  = 0;
};
#endif
