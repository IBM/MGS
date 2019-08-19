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

#include "EdgeCompCategoryBase.h"
#include "EdgePartitionItem.h"
#include "Simulation.h"

#include <cassert>
#include <vector>

EdgeCompCategoryBase::EdgeCompCategoryBase(
   Simulation& sim, const std::string& modelName)
   : CompCategoryBase(sim), EdgeType(), _partitions(0), _nbrPartitions(0),
     _modelName(modelName)
{
   sim.registerEdgeCompCat(this);
}

EdgeCompCategoryBase::~EdgeCompCategoryBase()
{
   delete[] _partitions;
//    ShallowArray<Edge*>::iterator it, end = _edgeList.end();
//    for (it = _edgeList.begin(); it != end; ++it) {
//       delete *it;
//    }
}

void EdgeCompCategoryBase::initPartitions(int numCores, int numGPUs)
{
   int totalSize = getNumOfEdges();
   if (numCores > totalSize) {
      numCores = totalSize;
   }

   if (numCores > 0) {
      _nbrPartitions = numCores;
      _partitions = new EdgePartitionItem[numCores];
      int chunkSize = int(floor(double(totalSize)/double(numCores)));

      int extraData = totalSize - (chunkSize * numCores);
      
      int startIndex = 0;
      for (int i = 0; i < numCores; ++i) {
	 _partitions[i].startIndex = startIndex;
	 _partitions[i].endIndex = startIndex + chunkSize - 1;
	 totalSize -= chunkSize;
	 if (i < extraData) {
	    _partitions[i].endIndex += 1;
	    totalSize--;
	 }
	 startIndex = _partitions[i].endIndex + 1;
      }
      assert(totalSize == 0);
   }
   getWorkUnits();
}
