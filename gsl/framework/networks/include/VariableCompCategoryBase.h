// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef VariableCompCategoryBase_H
#define VariableCompCategoryBase_H
#include "Copyright.h"

#include <memory>
#include <iostream>

#include "DistributableCompCategoryBase.h"
#include "VariableType.h"

class Variable;
class VariableDescriptor;
class ParameterSet;
class Simulation;
class WorkUnit;
class VariablePartitionItem;

class VariableCompCategoryBase : public DistributableCompCategoryBase, public VariableType
{

   public:
      VariableCompCategoryBase(Simulation& sim);
      virtual ~VariableCompCategoryBase();

      // VariableType functions to be implemented
#ifdef HAVE_MPI
      virtual void addToSendMap(int partitionId, Variable* var) = 0;
      virtual void allocateProxy(int fromPartitionId, VariableDescriptor* vd)=0;
#endif
      virtual VariableCompCategoryBase* getCompCategoryBase() {return this;}
      virtual void getInitializationParameterSet(std::unique_ptr<ParameterSet>& initPSet) = 0;
      virtual void getInAttrParameterSet(std::unique_ptr<ParameterSet>& inAttrPSet) = 0;
      virtual void getOutAttrParameterSet(std::unique_ptr<ParameterSet>& outAttrPSet) = 0;
      virtual void initPartitions(int numCores, int numGPUs);

      // move to CG later
      virtual void getQueriable(
	 std::unique_ptr<InstanceFactoryQueriable>&  dup);

   protected:

      VariablePartitionItem* _partitions;
      int _nbrPartitions;
};
#endif
