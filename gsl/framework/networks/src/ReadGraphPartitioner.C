// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifdef HAVE_MPI
#include <mpi.h>
#endif
#include "ReadGraphPartitioner.h"
#include <fstream>
#include <iostream>
#include <cassert>
#include "Granule.h"

ReadGraphPartitioner::ReadGraphPartitioner(const std::string& fileName, bool outputGraph)
   : Partitioner(), _fileName(fileName), _outputGraph(outputGraph)
{
}

void ReadGraphPartitioner::partition(std::vector<Granule*>& graph, 
				      unsigned numberOfPartitions)
{
  std::vector<Granule*>::iterator it, end = graph.end();
  
  std::ifstream* infile=new std::ifstream(_fileName.c_str());
  int pid=0;
  for (it = graph.begin(); it != end; ++it, ++pid) {
    (*infile) >> *(*it);
  }
  infile->close();
  delete infile;

  if (_outputGraph) {
    std::ofstream* outfile=new std::ofstream(_fileName.c_str());
    for (it = graph.begin(); it != end; ++it, ++pid) {
      (*outfile) << *(*it);
      (*outfile) << "\n";
    }
    delete outfile;
  }
}
