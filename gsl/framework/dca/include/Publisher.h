// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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

      virtual void duplicate(std::unique_ptr<Publisher>&& dup) const  = 0;
};
#endif
