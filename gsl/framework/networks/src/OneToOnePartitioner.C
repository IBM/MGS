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
