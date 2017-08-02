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

int VariableCompCategoryBase::initPartitions(int num)
{
   int n = _variableList.size();
   if (n==0) {
      num = 1;
      _nbrPartitions = num;
      _partitions = new VariablePartitionItem[num];
   
      _partitions[0].startIndex = 1;
      _partitions[0].endIndex =  0;
   }
   else{
      if (n < num) num = n;
      _nbrPartitions = num;
      _partitions = new VariablePartitionItem[num];
      int unitsPerPartition= n/num;
   
      for (int idx=0; idx < num; ++idx) {
        _partitions[idx].startIndex = idx*unitsPerPartition;
        _partitions[idx].endIndex =  (idx+1)*unitsPerPartition - 1;					
      }
       _partitions[num-1].endIndex = n - 1;
   }
   getWorkUnits();
   return num;
}

// move to CG later
void VariableCompCategoryBase::getQueriable(
   std::auto_ptr<InstanceFactoryQueriable>& dup)
{
   dup.reset(new InstanceFactoryQueriable(this));
   dup->setName(getModelName());
}

