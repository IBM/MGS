/*
=================================================================
Licensed Materials - Property of IBM

"Restricted Materials of IBM"

BMC-YKT-03-25-2018

(C) Copyright IBM Corp. 2005-2017  All rights reserved

US Government Users Restricted Rights -
Use, duplication or disclosure restricted by
GSA ADP Schedule Contract with IBM Corp.

================================================================
*/


#include "Lens.h"
#include "ChannelKAsCompCategory.h"
#include "NDPairList.h"
#include "CG_ChannelKAsCompCategory.h"

#include <mpi.h>
#include "NumberUtils.h"

ChannelKAsCompCategory::ChannelKAsCompCategory(Simulation& sim,
                                               const std::string& modelName,
                                               const NDPairList& ndpList)
    : CG_ChannelKAsCompCategory(sim, modelName, ndpList)
{
}

// GOAL:
//  1. compute Erev
//  2. find Q10 adjustment
void ChannelKAsCompCategory::computeE(RNG& rng)
{
  // step 1.
  if (getSharedMembers().T && getSharedMembers().K_EC &&
      getSharedMembers().K_IC)
  {
    dyn_var_t E_K;
    // E_rev  = RT/(zF)ln([K]o/[K]i)   [mV]
    E_K = 0.08617373 * *(getSharedMembers().T) *
          log(*(getSharedMembers().K_EC) / *(getSharedMembers().K_IC));
    getSharedMembers().E_K.push_back(E_K);
  }
#ifdef DEBUG
  std::cerr << getSimulation().getRank() << " : T=" << *getSharedMembers().T
            << " K_EC=" << *getSharedMembers().K_EC
            << " K_IC=" << *getSharedMembers().K_IC
            << " E_K=" << getSharedMembers().E_K[0] << std::endl;
#endif
  // Step 2. Find temperature adjustment factor Tadj
  //      based upon Q10 and T values
  // if (getSharedMembers().T and getSharedMembers().Tadj)
  if (getSharedMembers().T)
  {
    assert(*(getSharedMembers().T) > 273.15);
    getSharedMembers().Tadj = pow(
        Q10, ((*(getSharedMembers().T) - 273.15 - BASED_TEMPERATURE) / 10.0));
  }
}

//
// Return the statistics of distributing the instances of this nodetype
// onto different computing nodes
// i.e. the total instances,
//      the mean #-of-instance-being-processed by each node,
//      the stddev #-of-instance-being-processed by each node
void ChannelKAsCompCategory::count()
{
  long long totalCount, localCount = _nodes.size();
  MPI_Allreduce((void*)&localCount, (void*)&totalCount, 1, MPI_LONG_LONG,
                MPI_SUM, MPI_COMM_WORLD);
  float mean = (float(totalCount)) / getSimulation().getNumProcesses();
  float localVariance = Square(float(localCount) - mean);
  float totalVariance;
  MPI_Allreduce((void*)&localVariance, (void*)&totalVariance, 1, MPI_FLOAT,
                MPI_SUM, MPI_COMM_WORLD);
  float std = sqrt(totalVariance / getSimulation().getNumProcesses());

  if (getSimulation().getRank() == 0)
    printf("Total KAs Channel = %lld, Mean = %lf, StDev = %lf\n", totalCount,
           mean, std);
}
