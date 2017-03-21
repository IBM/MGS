// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef Partitioner_H
#define Partitioner_H
#include "Copyright.h"

//#include "Granule.h"
#include <vector>

class Granule;

class Partitioner
{
   public:
      Partitioner() {};
      virtual ~Partitioner(){};

      virtual void partition(std::vector<Granule*>& graph, 
			     unsigned numberOfPartitions) = 0;
      virtual bool requiresCostAggregation() = 0;
};

#endif
