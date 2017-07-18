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
