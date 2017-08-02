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

#ifndef ReadGraphPartitioner_H
#define ReadGraphPartitioner_H
#include "Copyright.h"

#include "Partitioner.h"

#include <string>
#include <vector>

class Granule;

class ReadGraphPartitioner : public Partitioner
{
   public:
      ReadGraphPartitioner(const std::string& fileName, bool outputGraph);
      virtual ~ReadGraphPartitioner(){};

      virtual void partition(std::vector<Granule*>& graph, 
			     unsigned numberOfPartitions);
      
      virtual bool requiresCostAggregation() {return false;}
      
   private:
      std::string _fileName;
      bool _outputGraph;
};

#endif
