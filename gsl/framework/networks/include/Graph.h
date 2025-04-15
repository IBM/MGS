// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef Graph_H
#define Graph_H
#include "Copyright.h"

#include <vector>


class Granule;
class Partitioner;

class Graph
{
   public:
      Graph(unsigned graphSize, unsigned numberOfPartitions);

      void setNode(unsigned graphNodeId, Granule* granule);

      unsigned getPartitionId(unsigned graphNodeId) const;
      void partition(Partitioner* partitioner);            

   private:
      std::vector<Granule*> _graphNodes;
      unsigned _numberOfPartitions;
};

#endif
