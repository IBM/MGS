// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
