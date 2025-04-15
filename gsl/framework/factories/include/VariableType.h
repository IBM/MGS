// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef VARIABLETYPE_H
#define VARIABLETYPE_H
#include "Copyright.h"

#include "InstanceFactory.h"
#include <memory>
#include <vector>
#include <string>
#include <map>
#include <cassert>
#include "Variable.h"
#include "DuplicatePointerArray.h"

class DataItem;
class InstanceFactoryQueriable;
class ConnectionIncrement;
class ParameterSet;
class NDPairList;
class Simulation;
class VariableCompCategoryBase;

class VariableType : public InstanceFactory
{
   public:
      VariableType();
      virtual Variable* allocateVariable()=0;
      virtual VariableCompCategoryBase* getCompCategoryBase() =0;
      virtual std::string getModelName() const =0;
      virtual void getInitializationParameterSet(std::unique_ptr<ParameterSet>& initPSet) = 0;
      virtual void getInAttrParameterSet(std::unique_ptr<ParameterSet>& inAttrPSet) = 0;
      virtual void getOutAttrParameterSet(std::unique_ptr<ParameterSet>& outAttrPSet) = 0;
      virtual ~VariableType();
      virtual void getInstance(std::unique_ptr<DataItem> & adi, 
			       std::vector<DataItem*> const * args, 
			       LensContext* c);
      virtual void getInstance(std::unique_ptr<DataItem> & adi, 
 			       const NDPairList& ndplist,
			       LensContext* c);
//      virtual ConnectionIncrement* getComputeCost() = 0;
     void setCreatingInstanceAtEachMPIProcess() { _instanceAtEachMPIProcess=true; };
   protected:
      DuplicatePointerArray<Variable> _variableList;
   private:
      inline void addGranuleToSimulation(
	 Simulation& sim, ConnectionIncrement* computeCost, unsigned variableIndex) const;
	  bool _instanceAtEachMPIProcess;
};
#endif
