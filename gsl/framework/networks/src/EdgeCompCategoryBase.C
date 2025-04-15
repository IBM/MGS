// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
