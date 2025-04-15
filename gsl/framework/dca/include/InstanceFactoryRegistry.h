// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef INSTANCEFACTORYREGISTRY_H
#define INSTANCEFACTORYREGISTRY_H
#include "Copyright.h"

#include <list>
#include <string>
#include <memory>


class InstanceFactory;
class InstanceFactoryRegistryQueriable;
class Simulation;
class DependencyParser;

class InstanceFactoryRegistry
{
   public:
      InstanceFactoryRegistry() {}
      virtual const std::list<InstanceFactory*> & getInstanceFactoryList() {
	 return _instanceFactoryList;
      }
      virtual InstanceFactory* getInstanceFactory(
	 Simulation& sim, DependencyParser& dep, 
	 std::string InstanceFactoryName) = 0;
      virtual void getQueriable(
	 std::unique_ptr<InstanceFactoryRegistryQueriable>& dup) = 0;
      virtual std::string getTypeName() {
	 return _typeName;
      }
      virtual ~InstanceFactoryRegistry() {}

   protected:
      std::list<InstanceFactory*> _instanceFactoryList;
      std::string _typeName;

};
#endif
