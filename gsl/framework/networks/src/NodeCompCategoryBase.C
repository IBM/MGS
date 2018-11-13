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
     _gridLayerDataArraySize(0), _corePartitions(0), _gpuPartitions(0),
     _nbrCorePartitions(0), _nbrGpuPartitions(0), _modelName(modelName)

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
   delete[] _corePartitions;
   delete[] _gpuPartitions;
}

void NodeCompCategoryBase::initPartitions(int numCores, int numGPUs)
{

   int n=getNbrComputationalUnits();
   if (n == 0) {
      numCores = numGPUs = 1;
      _nbrCorePartitions = _nbrGpuPartitions = numCores;
      _corePartitions = new NodePartitionItem[numCores];
      _gpuPartitions = new NodePartitionItem[numGPUs];
      
      _corePartitions[0].startIndex = _gpuPartitions[0].startIndex = 1;
      _corePartitions[0].endIndex = _gpuPartitions[0].endIndex = 0;
   } 
   else {
      if (n < numCores) numCores = n;
      if (n < numGPUs) numGPUs = n;
      _nbrCorePartitions = numCores;
      _nbrGpuPartitions = numGPUs;
      _corePartitions = new NodePartitionItem[numCores];
      _gpuPartitions = new NodePartitionItem[numGPUs];
      
      int unitsPerCorePartition= n/numCores;
      int unitsPerGpuPartition= n/numGPUs;
   
      for (int idx=0; idx < numCores; ++idx) {
        _corePartitions[idx].startIndex = idx*unitsPerCorePartition;
        _corePartitions[idx].endIndex =  (idx+1)*unitsPerCorePartition - 1;
      }
       _corePartitions[numCores-1].endIndex = n - 1;

      for (int idx=0; idx < numGPUs; ++idx) {
        _gpuPartitions[idx].startIndex = idx*unitsPerGpuPartition;
        _gpuPartitions[idx].endIndex =  (idx+1)*unitsPerGpuPartition - 1;
      }
       _gpuPartitions[numGPUs-1].endIndex = n - 1;
   }
   getWorkUnits();
}
