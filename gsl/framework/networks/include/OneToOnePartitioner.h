// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef OneToOnePartitioner_H
#define OneToOnePartitioner_H
#include "Copyright.h"

#include "Partitioner.h"

#include <string>
#include <vector>

class Granule;

class OneToOnePartitioner : public Partitioner
{
   public:
      OneToOnePartitioner(const std::string& fileName, bool outputGraph);
      virtual ~OneToOnePartitioner(){};

      virtual void partition(std::vector<Granule*>& graph, 
			     unsigned numberOfPartitions);
      
      virtual bool requiresCostAggregation() {return false;}
      
   private:
      std::string _fileName;
      bool _outputGraph;
      std::vector<int> _deck;
};

#endif
