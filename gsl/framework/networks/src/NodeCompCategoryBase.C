// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "NodeCompCategoryBase.h"
//#include "ParameterSet.h"
//#include "NodeAccessor.h"
//#include "GridLayerDescriptor.h"
#include "GridLayerData.h"
#include "Simulation.h"
//#include "WorkUnit.h"
#include "NodePartitionItem.h"
//#include "Node.h"

#include <cassert>
#include <vector>

NodeCompCategoryBase::NodeCompCategoryBase(
   Simulation& sim, const std::string& modelName)
   : DistributableCompCategoryBase(sim), NodeType(), _gridLayerDataArray(0), 
     _gridLayerDataArraySize(0), _CPUpartitions(0), _GPUpartitions(0),
     _nbrCPUpartitions(0), _nbrGPUpartitions(0), _modelName(modelName)

{
  _gridLayerDataOffsets.push_back(0);
}

NodeCompCategoryBase::~NodeCompCategoryBase()
{

   std::deque<GridLayerData*>::iterator it, end = _gridLayerDataList.end();
   for(it = _gridLayerDataList.begin(); it != end; ++it) {
       delete *it;
   }
   delete[] _gridLayerDataArray;
   delete[] _CPUpartitions;
   delete[] _GPUpartitions;
}

void NodeCompCategoryBase::initPartitions(int numCpuWorkUnits, int numGpuWorkUnits)
{
   _nbrCPUpartitions = numCpuWorkUnits;
   _nbrGPUpartitions = numGpuWorkUnits;

   int n=getNbrComputationalUnits();
   if (n == 0) {
      _nbrCPUpartitions = _nbrGPUpartitions = 1;
      _CPUpartitions = new NodePartitionItem[_nbrCPUpartitions];
      _GPUpartitions = new NodePartitionItem[_nbrGPUpartitions];
      
      _CPUpartitions[0].startIndex = _GPUpartitions[0].startIndex = 1;
      _CPUpartitions[0].endIndex = _GPUpartitions[0].endIndex = 0;
   } 
   else {
      if (n < numCpuWorkUnits) _nbrCPUpartitions = n;
      if (n < numGpuWorkUnits) _nbrGPUpartitions = n;
      _CPUpartitions = new NodePartitionItem[_nbrCPUpartitions];
      _GPUpartitions = new NodePartitionItem[_nbrGPUpartitions];
      
      int unitsPerCpuPartition= n/_nbrCPUpartitions;
      for (int idx=0; idx < _nbrCPUpartitions; ++idx) {
        _CPUpartitions[idx].startIndex = idx*unitsPerCpuPartition;
        _CPUpartitions[idx].endIndex =  (idx+1)*unitsPerCpuPartition - 1;
      }
       _CPUpartitions[_nbrCPUpartitions-1].endIndex = n - 1;

      int unitsPerGpuPartition= n/_nbrGPUpartitions;   
      for (int idx=0; idx < _nbrGPUpartitions; ++idx) {
        _GPUpartitions[idx].startIndex = idx*unitsPerGpuPartition;
        _GPUpartitions[idx].endIndex =  (idx+1)*unitsPerGpuPartition - 1;
      }
       _GPUpartitions[_nbrGPUpartitions-1].endIndex = n - 1;
   }
   getWorkUnits();
}
