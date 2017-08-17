// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

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
      virtual void getInitializationParameterSet(std::auto_ptr<ParameterSet>& initPSet) = 0;
      virtual void getInAttrParameterSet(std::auto_ptr<ParameterSet>& inAttrPSet) = 0;
      virtual void getOutAttrParameterSet(std::auto_ptr<ParameterSet>& outAttrPSet) = 0;
      virtual int initPartitions(int num);

      // move to CG later
      virtual void getQueriable(
	 std::auto_ptr<InstanceFactoryQueriable>&  dup);

   protected:

      VariablePartitionItem* _partitions;
      int _nbrPartitions;
};
#endif
