// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef BGCartesianPartitioner_H
#define BGCartesianPartitioner_H
#include "Copyright.h"

#include "Partitioner.h"
#include "Simulation.h"

#include <string>
#include <vector>
#define DIM 3

class Granule;

class BGCartesianPartitioner : public Partitioner
{
   public:
      BGCartesianPartitioner(const std::string& fileName, bool outputGraph, Simulation* sim);
      virtual ~BGCartesianPartitioner(){};

      virtual void partition(std::vector<Granule*>& graph, 
			     unsigned numberOfPartitions);
      
      virtual bool requiresCostAggregation() {return false;}
      
   private:
      std::string _fileName;
      bool _outputGraph;
      Simulation* _sim;
      int _nprocs;
      int _mesh[DIM];
      std::vector<int> _meshOrder;
};

#endif
