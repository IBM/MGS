// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#include "Mgs.h"
#include "ChannelKDR_AISCompCategory.h"
#include "NDPairList.h"
#include "CG_ChannelKDR_AISCompCategory.h"
#include <math.h>

#include <mpi.h>
#include "NumberUtils.h"

ChannelKDR_AISCompCategory::ChannelKDR_AISCompCategory(Simulation& sim, 
                                                       const std::string& modelName, 
                                                       const NDPairList& ndpList) 
   : CG_ChannelKDR_AISCompCategory(sim, modelName, ndpList)
{
}

void ChannelKDR_AISCompCategory::computeE(RNG& rng) 
{
// step 1.
  if (getSharedMembers().T && getSharedMembers().K_EC &&
      getSharedMembers().K_IC)
  {
    dyn_var_t E_K;
    // E_rev  = RT/(zF)ln([Na]o/[Na]i)   [mV]
    E_K = 0.08686 * *(getSharedMembers().T) *
          log(*(getSharedMembers().K_EC) / *(getSharedMembers().K_IC));
    getSharedMembers().E_K.push_back(E_K);
#ifdef DEBUG_HH
    std::cerr << getSimulation().getRank() << " : T=" << *getSharedMembers().T
              << " K_EC=" << *getSharedMembers().K_EC << " K_IC=";
    std::cerr << *getSharedMembers().K_IC
              << " E_K=" << getSharedMembers().E_K[0] << std::endl;
#endif
  }
  // Step 2. Find temperature adjustment factor Tadj
  //      based upon Q10 and T values
  // if (getSharedMembers().T and getSharedMembers().Tadj)
  if (getSharedMembers().T)
  {
    assert(*(getSharedMembers().T) > 273.15);
    getSharedMembers().Tadj = pow(
        Q10, ((*(getSharedMembers().T) - 273.15 - BASED_TEMPERATURE) / 10.0));
  }
  // pow(static_cast<dyn_var_t>(Q10), ((*(getSharedMembers().T) - 273.15 -
  // BASED_TEMPERATURE) / 10.0));
  //(((*(getSharedMembers().T) - 273.15 - BASED_TEMPERATURE) / 10.0));
}

//
// Return the statistics of distributing the instances of this nodetype
// onto different computing nodes
// i.e. the total instances,
//      the mean #-of-instance-being-processed by each node,
//      the stddev #-of-instance-being-processed by each node
void ChannelKDR_AISCompCategory::count()
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
    printf("Total KDR_AIS Channel = %lld, Mean = %lf, StDev = %lf\n", totalCount,
           mean, std);
}
