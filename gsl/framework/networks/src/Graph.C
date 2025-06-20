// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifdef HAVE_MPI
#include <mpi.h>
#endif
#include "Graph.h"
#include "Partitioner.h"
#include "Granule.h"
#include <cassert>
#include <iostream>
#include <cstdlib>

Graph::Graph(unsigned graphSize, unsigned numberOfPartitions)
   : _numberOfPartitions(numberOfPartitions)
{
   _graphNodes.resize(graphSize);
   for (unsigned i=0; i<graphSize; ++i) _graphNodes[i] = 0;
}

unsigned Graph::getPartitionId(unsigned graphNodeId) const
{
   unsigned retval = 0;
   assert(graphNodeId < _graphNodes.size());
   if (_graphNodes[graphNodeId] != 0){
      retval = _graphNodes[graphNodeId]->getPartitionId();
   }
   else {
      std::cerr <<"Attempt to access partition id from null node pointer on graph "<<std::endl;
      exit(-1);
   }
   return retval;
}


void Graph::partition(Partitioner* partitioner)
{
   partitioner->partition(_graphNodes, _numberOfPartitions);
}

void Graph::setNode(unsigned graphNodeId, Granule* granule) 
{
   assert(graphNodeId < _graphNodes.size());
   _graphNodes[graphNodeId] = granule;
}
