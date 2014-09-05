// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

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
