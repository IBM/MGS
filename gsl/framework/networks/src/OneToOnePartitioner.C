// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifdef HAVE_MPI

#define STRIDE 8
#define STRIDE2 16
#include <mpi.h>
#endif
#include "OneToOnePartitioner.h"
#include <fstream>
#include <iostream>
#include <cassert>
#include "Granule.h"
#include "RNG.h"
#include "rndm.h"


OneToOnePartitioner::OneToOnePartitioner(const std::string& fileName, bool outputGraph)
   : Partitioner(), _fileName(fileName), _outputGraph(outputGraph)
{
}

void OneToOnePartitioner::partition(std::vector<Granule*>& graph, 
				      unsigned numberOfPartitions)
{
  std::vector<Granule*>::iterator it, end = graph.end();
  
  std::ofstream* file=0;
  if (_outputGraph) file=new std::ofstream(_fileName.c_str());
  int gid=0;
  for (it = graph.begin(); it != end; ++it, ++gid) {
    (*it)->setPartitionId(gid%numberOfPartitions);
    if (_outputGraph) {
      (*file) << *(*it);
      (*file) << "\n";
    }
  }
  if (_outputGraph) file->close();
  delete file;
}
