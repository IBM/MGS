// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "VariableCompCategoryBase.h"
#include "VariablePartitionItem.h"
#include "InstanceFactoryQueriable.h"

#include <cassert>
#include <vector>

VariableCompCategoryBase::VariableCompCategoryBase(
   Simulation& sim)
   : DistributableCompCategoryBase(sim), VariableType(), _partitions(0), _nbrPartitions(0)
{
}

VariableCompCategoryBase::~VariableCompCategoryBase()
{
   delete[] _partitions;
}

void VariableCompCategoryBase::initPartitions(int numCores, int numGPUs)
{
   int n = _variableList.size();
   if (n==0) {
      numCores = 1;
      _nbrPartitions = numCores;
      _partitions = new VariablePartitionItem[numCores];
   
      _partitions[0].startIndex = 1;
      _partitions[0].endIndex =  0;
   }
   else{
      if (n < numCores) numCores = n;
      _nbrPartitions = numCores;
      _partitions = new VariablePartitionItem[numCores];
      int unitsPerPartition= n/numCores;
   
      for (int idx=0; idx < numCores; ++idx) {
        _partitions[idx].startIndex = idx*unitsPerPartition;
        _partitions[idx].endIndex =  (idx+1)*unitsPerPartition - 1;					
      }
       _partitions[numCores-1].endIndex = n - 1;
   }
   getWorkUnits();
}

// move to CG later
void VariableCompCategoryBase::getQueriable(
   std::unique_ptr<InstanceFactoryQueriable>& dup)
{
   dup.reset(new InstanceFactoryQueriable(this));
   dup->setName(getModelName());
}

