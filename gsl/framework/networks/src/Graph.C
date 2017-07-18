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
