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

#ifndef TYPEMANAGER_H
#define TYPEMANAGER_H
#include "Copyright.h"

#include <string>
#include <map>

#include "FactoryMap.h"

class Simulation;
class DependencyParser;

template <class T>
class TypeManager
{
   public:
      TypeManager();
      T* getType(Simulation& sim, DependencyParser& dep,
		 const std::string& modelName, const NDPairList& ndpList);
      ~TypeManager();

   private:
      std::map<std::string, T*> _interfaceMap;

};


template <class T>
TypeManager<T>::TypeManager()
{
}

template <class T>
T* TypeManager<T>::getType(Simulation& sim, DependencyParser& dep,
			   const std::string& modelName, 
			   const NDPairList& ndpList)
{
   T* ptrType=0;
   typename std::map<std::string, T*>::iterator 
      iter = _interfaceMap.find(modelName);

   if (iter!=_interfaceMap.end())
      ptrType = (*iter).second;
   else {
      ptrType = FactoryMap<T>::getFactoryMap()->load(sim, dep, modelName, 
						     ndpList);
      _interfaceMap[modelName] = ptrType;
   }
   return ptrType;
}

template <class T>
TypeManager<T>::~TypeManager()
{
   typename std::map<std::string, T*>::iterator it, end = _interfaceMap.end();
   for (it = _interfaceMap.begin(); it != end; ++it) {
      delete it->second;
   }
}

#endif
