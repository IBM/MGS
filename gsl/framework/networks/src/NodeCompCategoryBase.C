// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
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
     _gridLayerDataArraySize(0), _partitions(0), _nbrPartitions(0),
   _modelName(modelName)

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
   delete[] _partitions;
}

int NodeCompCategoryBase::initPartitions(int num)
{

   int n=getNbrComputationalUnits();
   if (n == 0){
      num = 1;
      _nbrPartitions = num;
      _partitions = new NodePartitionItem[num];
   
      _partitions[0].startIndex = 1;
      _partitions[0].endIndex =  0;
   } 
   else{
      if (n < num) num = n;
      _nbrPartitions = num;
      _partitions = new NodePartitionItem[num];
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
