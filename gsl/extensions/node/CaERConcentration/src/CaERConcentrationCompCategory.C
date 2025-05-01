// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Mgs.h"
#include "CaERConcentrationCompCategory.h"
#include "NDPairList.h"
#include "CG_CaERConcentrationCompCategory.h"
#include <math.h>
//#define DEBUG_HH

CaERConcentrationCompCategory::CaERConcentrationCompCategory(
    Simulation& sim, const std::string& modelName, const NDPairList& ndpList)
    : CG_CaERConcentrationCompCategory(sim, modelName, ndpList)
{
}

//GOAL: get any derived parameters
//  1. the bmt = 1/(beta * (dt/2))
//     beta = bufferingFactor = fast-buffering factor [Wagner-Keizer,1994]
//     beta = 1/(1 + [Bm]total * Km / (Km + Caer)^2 + [Bs]total * Ks / (Ks + Caer)^2)
//     NOTE: We assume no mobile buffer in ER
void CaERConcentrationCompCategory::deriveParameters(RNG& rng)
{
  if (getSharedMembers().deltaT)
  {
#if CALCIUM_ER_DYNAMICS == FAST_BUFFERING
    getSharedMembers().bmt =
        2.0 / (getSharedMembers().beta * *(getSharedMembers().deltaT));
#elif CALCIUM_ER_DYNAMICS == REGULAR_DYNAMICS
    getSharedMembers().bmt = 2.0 / (*(getSharedMembers().deltaT)) ;
#else
    assert(0);
#endif
    assert(getSharedMembers().bmt > 0);

#ifdef DEBUG_HH
    //std::cerr << getSimulation().getRank()
    //          << " : CaERConcentrations : " << _nodes.size() << " [ ";
    //for (int i = 0; i < _nodes.size(); ++i)
    //  std::cerr << _nodes[i].getSize() << " ";
    //std::cerr << " ]" << std::endl;
#endif
  }
}

// Return the statistics of distributing the instances of this nodetype
// onto different computing nodes
// i.e. the total instances,
//      the mean #-of-instance-being-processed by each node,
//      the stddev #-of-instance-being-processed by each node
void CaERConcentrationCompCategory::count()
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
    printf("Total CaERConcentration = %lld, Mean = %lf, StDev = %lf\n",
           totalCount, mean, std);

  ////

  totalCount = localCount = 0;
  ShallowArray<CaERConcentration>::iterator nodesIter = _nodes.begin(),
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
    printf("Total CaER Compartment = %lld, Mean = %lf, StDev = %lf\n",
           totalCount, mean, std);
}
