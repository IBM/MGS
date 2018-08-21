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
	 std::auto_ptr<InstanceFactoryRegistryQueriable>& dup) = 0;
      virtual std::string getTypeName() {
	 return _typeName;
      }
      virtual ~InstanceFactoryRegistry() {}

   protected:
      std::list<InstanceFactory*> _instanceFactoryList;
      std::string _typeName;

};
#endif
