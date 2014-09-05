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
