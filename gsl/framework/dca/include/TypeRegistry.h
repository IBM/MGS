// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef TYPEREGISTRY_H
#define TYPEREGISTRY_H
#include "Copyright.h"

#include <memory>
#include <list>
#include <string>
#include <sstream>

#include "InstanceFactoryRegistry.h"
#include "TypeManager.h"
#include "TypeClassifier.h"
#include "InstanceFactoryRegistryQueriable.h"
#include "ConnectionIncrement.h"

class Simulation;
class DependencyParser;
class InstanceFactory;
class NDPairList;

template <class T>
class TypeRegistry : public InstanceFactoryRegistry
{
   public:
      TypeRegistry();
      T* getType(Simulation& sim, DependencyParser& dep, std::string typeName);
      InstanceFactory* getInstanceFactory(
	 Simulation& sim, DependencyParser& dep, 
	 std::string instanceFactoryName);
      virtual void getQueriable(
	 std::unique_ptr<InstanceFactoryRegistryQueriable>& dup);

   private:
      TypeManager<T> _typeManager;
      NDPairList _dummyNDPList;
};

template <class T>
TypeRegistry<T>::TypeRegistry()
{
   _typeName = TypeClassifier<T>::getName();
}

template <class T>
void TypeRegistry<T>::getQueriable(
   std::unique_ptr<InstanceFactoryRegistryQueriable>& dup)
{
   dup.reset(new InstanceFactoryRegistryQueriable(this));
}

template <class T>
T* TypeRegistry<T>::getType(Simulation& sim, DependencyParser& dep, 
			    std::string typeName)
{
   bool insert = true;

   T* rval = 0;
   typename std::list<InstanceFactory*>::iterator it, 
      end = _instanceFactoryList.end();
   for (it = _instanceFactoryList.begin(); it != end; ++it) {
      if (typeName == (*it)->getName()) {
         insert = false;
	 break;
      }
   }
   
   rval = _typeManager.getType(sim, dep, typeName, _dummyNDPList);

   if (insert && (rval != 0)) {
      //If the pointer is null, then the typemanager failed to load it
      _instanceFactoryList.push_back(rval);
   }
   return rval;
}

template <class T>
InstanceFactory* TypeRegistry<T>::getInstanceFactory(
   Simulation& sim, DependencyParser& dep, std::string instanceFactoryName)
{
   return dynamic_cast<InstanceFactory*>(getType(sim, dep, instanceFactoryName));
}

#endif
