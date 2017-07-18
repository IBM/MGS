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
