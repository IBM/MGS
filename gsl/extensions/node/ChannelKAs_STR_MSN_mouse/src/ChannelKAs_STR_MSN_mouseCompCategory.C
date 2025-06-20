// =============================================================================
// (C) Copyright 2018 New Jersey Institute of Technology. All rights reserved.
//
// =============================================================================

#include "Mgs.h"
#include "ChannelKAs_STR_MSN_mouseCompCategory.h"
#include "NDPairList.h"
#include "CG_ChannelKAs_STR_MSN_mouseCompCategory.h"

#include <mpi.h>
#include "NumberUtils.h"

ChannelKAs_STR_MSN_mouseCompCategory::ChannelKAs_STR_MSN_mouseCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
   : CG_ChannelKAs_STR_MSN_mouseCompCategory(sim, modelName, ndpList)
{
}

// GOAL:
//  1. compute Erev
//  2. find Q10 adjustment
void ChannelKAs_STR_MSN_mouseCompCategory::computeE(RNG& rng) 
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
void ChannelKAs_STR_MSN_mouseCompCategory::count()
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
    printf("Total KAs STR MSN mouse Channel = %lld, Mean = %lf, StDev = %lf\n", totalCount,
           mean, std);
}
