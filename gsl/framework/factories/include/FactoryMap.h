// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
// The FactoryMap is a templatized class that dynamically
// loads the parameter type. It also returns the loaded type.

#ifndef FACTORYMAP_H
#define FACTORYMAP_H
#include "Copyright.h"
#include <map>
#include <string>
#include <iostream>
#include <stdio.h>

#include "LoaderException.h"
#include "TypeClassifier.h"
#include "NDPairList.h"
#include "DependencyParser.h"

class Simulation;

template <class T>
class FactoryMap
{
   public:
      // Type definition for the pointer to a function.
      typedef T* (*PtrToFunction)(Simulation&, const NDPairList&);
      typedef std::map<std::string, PtrToFunction> MapType;

      T* load(Simulation& s, DependencyParser& d, const std::string& modelName,
	      const NDPairList& ndpList);
      // If we didn't use PtrToFunction, we'd use the following as definition.
      //      inline void addFactory(std::string modelName, 
      //                  T* (*factory)(Simulation &, const NDPairList&) );
      inline void addFactory(std::string modelName, PtrToFunction factory);
      inline static FactoryMap<T>* getFactoryMap();
      ~FactoryMap();

      static FactoryMap<T>* _factoryMap;
      std::vector<std::string> getNamesOfSupportedType()
      {
	 std::vector<std::string> v;
	 for(auto it = _fmap.begin(); it != _fmap.end(); ) {
	    v.push_back(it->first);
	    std::cout << it->first ;
	    if ((++it) != _fmap.end()) 
	       std::cout << "; ";
	    else
	       std::cout << ".\n";
	 }
	 return v;
      }

   private:
      MapType _fmap;
      FactoryMap();
      FactoryMap(FactoryMap & fm);
};

template <class T>
FactoryMap<T>::FactoryMap()
{
}

template <class T>
FactoryMap<T>::FactoryMap(FactoryMap & nfm)
{
   _fmap = nfm._fmap;
}

template <class T>
T* FactoryMap<T>::load(Simulation& s, DependencyParser& d, 
		       const std::string& modelName, const NDPairList& ndpList)
{
   try
   {
      typename MapType::iterator iter;
      iter = _fmap.find(modelName);
      if (iter == _fmap.end()) {
	 if (d.load(modelName))
	    iter = _fmap.find(modelName);
         else
            return 0;
      }
      return (T*)((*iter).second)(s, ndpList);

   }
   catch (LoaderException e) {
      std::cerr<<e.getError().c_str()<<std::endl;
      return 0;
   }
}

template <class T>
void FactoryMap<T>::addFactory(std::string modelName, PtrToFunction factory)
{
   if (factory == 0) 
      return;
   // Can't use cout because of some odd error in AIX.
   // std::cout << "Adding "<<modelName<<" to " << 
   // TypeClassifier<T>::getName() << " FactoryMap." << std::endl;
#ifdef VERBOSE
   printf("Adding %s to %s FactoryMap.\n", modelName.c_str(), 
	  TypeClassifier<T>::getName().c_str());
#endif
   getFactoryMap()->_fmap[modelName] = factory;
}

template <class T>
FactoryMap<T>* FactoryMap<T>::getFactoryMap()
{
   if (_factoryMap == 0) _factoryMap = new FactoryMap<T>();
   return _factoryMap;
}

#ifdef AIX
// Workaround of a gcc compiler bug in AIX.
class ConstantType;
class NodeType;
class EdgeType;
class FunctorType;
class TriggerType;
class StructType;
class VariableType;

extern FactoryMap<ConstantType>* _GlobalConstantTypeFactoryMap;
extern FactoryMap<NodeType>* _GlobalNodeTypeFactoryMap;
extern FactoryMap<EdgeType>* _GlobalEdgeTypeFactoryMap;
extern FactoryMap<FunctorType>* _GlobalFunctorTypeFactoryMap;
extern FactoryMap<TriggerType>* _GlobalTriggerTypeFactoryMap;
extern FactoryMap<StructType>* _GlobalStructTypeFactoryMap;
extern FactoryMap<VariableType>* _GlobalVariableTypeFactoryMap;

template <>
FactoryMap<ConstantType>* FactoryMap<ConstantType>::getFactoryMap()
{
   if (_GlobalConstantTypeFactoryMap == 0) 
      _GlobalConstantTypeFactoryMap = new FactoryMap<ConstantType>();
   return _GlobalConstantTypeFactoryMap;
}

template <>
FactoryMap<NodeType>* FactoryMap<NodeType>::getFactoryMap()
{
   if (_GlobalNodeTypeFactoryMap == 0) 
      _GlobalNodeTypeFactoryMap = new FactoryMap<NodeType>();
   return _GlobalNodeTypeFactoryMap;
}

template <>
FactoryMap<EdgeType>* FactoryMap<EdgeType>::getFactoryMap()
{
   if (_GlobalEdgeTypeFactoryMap == 0) 
      _GlobalEdgeTypeFactoryMap = new FactoryMap<EdgeType>();
   return _GlobalEdgeTypeFactoryMap;
}

template <>
FactoryMap<FunctorType>* FactoryMap<FunctorType>::getFactoryMap()
{
   if (_GlobalFunctorTypeFactoryMap == 0) 
      _GlobalFunctorTypeFactoryMap = new FactoryMap<FunctorType>();
   return _GlobalFunctorTypeFactoryMap;
}

template <>
FactoryMap<TriggerType>* FactoryMap<TriggerType>::getFactoryMap()
{
   if (_GlobalTriggerTypeFactoryMap == 0) 
      _GlobalTriggerTypeFactoryMap = new FactoryMap<TriggerType>();
   return _GlobalTriggerTypeFactoryMap;
}

template <>
FactoryMap<StructType>* FactoryMap<StructType>::getFactoryMap()
{
   if (_GlobalStructTypeFactoryMap == 0) 
      _GlobalStructTypeFactoryMap = new FactoryMap<StructType>();
   return _GlobalStructTypeFactoryMap;
}

template <>
FactoryMap<VariableType>* FactoryMap<VariableType>::getFactoryMap()
{
   if (_GlobalVariableTypeFactoryMap == 0) 
      _GlobalVariableTypeFactoryMap = new FactoryMap<VariableType>();
   return _GlobalVariableTypeFactoryMap;
}

#endif // AIX workaround

template <class T>
FactoryMap<T>::~FactoryMap()
{
   delete _factoryMap;
}

template <class T>
FactoryMap<T>* FactoryMap<T>::_factoryMap = 0;

#endif
