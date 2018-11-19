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
     _gridLayerDataArraySize(0), _cpuPartitions(0), _gpuPartitions(0),
     _nbrCpuPartitions(0), _nbrGpuPartitions(0), _modelName(modelName)

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
   delete[] _cpuPartitions;
   delete[] _gpuPartitions;
}

void NodeCompCategoryBase::initPartitions(int numCpuWorkUnits, int numGpuWorkUnits)
{
   _nbrCpuPartitions = numCpuWorkUnits;
   _nbrGpuPartitions = numGpuWorkUnits;

   int n=getNbrComputationalUnits();
   if (n == 0) {
      _nbrCpuPartitions = _nbrGpuPartitions = 1;
      _cpuPartitions = new NodePartitionItem[_nbrCpuPartitions];
      _gpuPartitions = new NodePartitionItem[_nbrGpuPartitions];
      
      _cpuPartitions[0].startIndex = _gpuPartitions[0].startIndex = 1;
      _cpuPartitions[0].endIndex = _gpuPartitions[0].endIndex = 0;
   } 
   else {
      if (n < numCpuWorkUnits) _nbrCpuPartitions = n;
      if (n < numGpuWorkUnits) _nbrGpuPartitions = n;
      _cpuPartitions = new NodePartitionItem[_nbrCpuPartitions];
      _gpuPartitions = new NodePartitionItem[_nbrGpuPartitions];
      
      int unitsPerCpuPartition= n/_nbrCpuPartitions;
      for (int idx=0; idx < _nbrCpuPartitions; ++idx) {
        _cpuPartitions[idx].startIndex = idx*unitsPerCpuPartition;
        _cpuPartitions[idx].endIndex =  (idx+1)*unitsPerCpuPartition - 1;
      }
       _cpuPartitions[_nbrCpuPartitions-1].endIndex = n - 1;

      int unitsPerGpuPartition= n/_nbrGpuPartitions;   
      for (int idx=0; idx < _nbrGpuPartitions; ++idx) {
        _gpuPartitions[idx].startIndex = idx*unitsPerGpuPartition;
        _gpuPartitions[idx].endIndex =  (idx+1)*unitsPerGpuPartition - 1;
      }
       _gpuPartitions[_nbrGpuPartitions-1].endIndex = n - 1;
   }
   getWorkUnits();
}
