// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "NodeSetSPMVariable.h"
#include "Simulation.h"
#include "CG_NodeSetSPMVariable.h"
#include <memory>
#include <fstream>

#ifdef HAVE_MPI
#include <mpi.h>
#endif
#include <iostream>
#include <cstdio>

//CUDA_CALLABLE 
void NodeSetSPMVariable::initialize(RNG& rng) 
{
   assert(coords.size() == vals.size());
   dimx = dimy = 0;
   Array<CoordsStruct>::iterator it, end = coords.end();
   for (it = coords.begin(); it != end; ++it) {
     if (it->coords[1] > dimx) dimx = it->coords[1];
     if (it->coords[0] > dimy) dimy = it->coords[0];
   }
   ++dimx;
   ++dimy;
}

void NodeSetSPMVariable::dca(Trigger* trigger, NDPairList* ndPairList) 
{
  /*
#ifdef HAVE_MPI
   int mPid;
   MPI_Comm_rank(MPI_COMM_WORLD, &mPid);
   std::cerr << "NodeSetSPMVariable::dca(), begin, my space id=" << mPid << std::endl;
   printf("NodeSetSPMVariable::dca(), begin, my space id=%d\n", mPid);
#endif
  */
   unsigned iteration = getSimulation().getIteration();
   std::ostringstream outfileName;
   outfileName<<fileName<<"_"<<iteration<<".spm";
   FILE* outfile = fopen(outfileName.str().c_str(), "wt");
   if (outfile==0) {
     std::cerr<<"Could not open file \""<<outfileName.str()<<"\" !"<<std::endl;
     exit(0);
   }
   fprintf(outfile, "%d\t%d\n", dimx, dimy);
   Array<CoordsStruct>::iterator it, end = coords.end();
   it = coords.begin();
   Array<float*>::iterator it2, end2 = vals.end();
   it2 = vals.begin();
   for (; it != end; ++it, ++it2) {
     fprintf(outfile, "%d\t%d\t%f\n", it->coords[0]+1, it->coords[1]+1, **it2);
   }
   fclose(outfile);
   /*
#ifdef HAVE_MPI
   std::cerr << "NodeSetSPMVariable::dca(), finished, my space id=" << mPid << std::endl;
   printf("NodeSetSPMVariable::dca(), finished, my space id=%d\n", mPid);
#endif
   */
//   exit(99);
}

NodeSetSPMVariable::NodeSetSPMVariable() 
   : CG_NodeSetSPMVariable()
{
}

NodeSetSPMVariable::~NodeSetSPMVariable() 
{
}

void NodeSetSPMVariable::duplicate(std::unique_ptr<NodeSetSPMVariable>& dup) const
{
   dup.reset(new NodeSetSPMVariable(*this));
}

void NodeSetSPMVariable::duplicate(std::unique_ptr<Variable>& dup) const
{
   dup.reset(new NodeSetSPMVariable(*this));
}

void NodeSetSPMVariable::duplicate(std::unique_ptr<CG_NodeSetSPMVariable>& dup) const
{
   dup.reset(new NodeSetSPMVariable(*this));
}

