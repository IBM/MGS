// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "Lens.h"
#include "CaConcentrationCompCategory.h"
#include "NDPairList.h"
#include "CG_CaConcentrationCompCategory.h"
#include <math.h>
#include <mpi.h>
//#define DEBUG_HH

CaConcentrationCompCategory::CaConcentrationCompCategory(
    Simulation& sim, const std::string& modelName, const NDPairList& ndpList)
    : CG_CaConcentrationCompCategory(sim, modelName, ndpList)
{
}

// GOAL: get any derived parameters
//  1. the bmt = 1/(beta * (dt/2))
//     beta = bufferingFactor = fast-buffering factor [Wagner-Keizer,1994]
//     beta = 1/(1 + [Bm]total * Km / (Km + Cacyto)^2 + [Bs]total * Ks / (Ks +
//     Cacyto)^2)
void CaConcentrationCompCategory::deriveParameters(RNG& rng)
{
  if (getSharedMembers().deltaT)
  {
#if CALCIUM_CYTO_DYNAMICS == FAST_BUFFERING
    getSharedMembers().bmt =
        2.0 / (getSharedMembers().beta * *(getSharedMembers().deltaT));
#else
    assert(0);
#endif

#ifdef DEBUG_HH
// std::cerr << getSimulation().getRank()
//          << " : CaConcentrations : " << _nodes.size() << " [ ";
// for (int i = 0; i < _nodes.size(); ++i)
//  std::cerr << _nodes[i].getSize() << " ";
// std::cerr << " ]" << std::endl;
#endif
  }
}

// Return the statistics of distributing the instances of this nodetype
// onto different computing nodes
// i.e. the total instances,
//      the mean #-of-instance-being-processed by each node,
//      the stddev #-of-instance-being-processed by each node
void CaConcentrationCompCategory::count()
{
  long long totalCount, localCount = _nodes.size();
  MPI_Allreduce((void*)&localCount, (void*)&totalCount, 1, MPI_LONG_LONG,
                MPI_SUM, MPI_COMM_WORLD);
  float localVar, totalVar,
      mean = float(totalCount) / getSimulation().getNumProcesses();
  localVar = (float(localCount) - mean) * (float(localCount) - mean);
  MPI_Allreduce((void*)&localVar, (void*)&totalVar, 1, MPI_FLOAT, MPI_SUM,
                MPI_COMM_WORLD);
  float std = sqrt(totalVar / getSimulation().getNumProcesses());
  if (getSimulation().getRank() == 0)
    printf("Total CaConcentration (ComputeBranches) = %lld, Mean = %lf, StDev = %lf\n",
           totalCount, mean, std);

  ////

  totalCount = localCount = 0;
  ShallowArray<CaConcentration>::iterator nodesIter = _nodes.begin(),
                                          nodesEnd = _nodes.end();
  for (; nodesIter != nodesEnd; ++nodesIter)
  {
    localCount += nodesIter->dimensions.size();
  }

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Allreduce((void*)&localCount, (void*)&totalCount, 1, MPI_LONG_LONG,
                MPI_SUM, MPI_COMM_WORLD);
  mean = totalCount / getSimulation().getNumProcesses();
  localVar = (float(localCount) - mean) * (float(localCount) - mean);
  MPI_Allreduce((void*)&localVar, (void*)&totalVar, 1, MPI_FLOAT, MPI_SUM,
                MPI_COMM_WORLD);
  std = sqrt(totalVar / getSimulation().getNumProcesses());
  if (getSimulation().getRank() == 0)
    printf("Total Ca Compartment = %lld, Mean = %lf, StDev = %lf\n", totalCount,
           mean, std);
}
