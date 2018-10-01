//=================================================================
//Licensed Materials - Property of IBM
//
//"Restricted Materials of IBM"
//
//BMC-YKT-03-25-2018
//
//(C) Copyright IBM Corp. 2005-2017  All rights reserved
//
//US Government Users Restricted Rights -
//Use, duplication or disclosure restricted by
//GSA ADP Schedule Contract with IBM Corp.
//
//================================================================
#include "Lens.h"
#include "ChannelCaHVACompCategory.h"
#include "NDPairList.h"
#include "CG_ChannelCaHVACompCategory.h"

#include <mpi.h>
#include "NumberUtils.h"
#include "NTSMacros.h"

ChannelCaHVACompCategory::ChannelCaHVACompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
   : CG_ChannelCaHVACompCategory(sim, modelName, ndpList)
{
}

// GOAL:
//  2. find Q10 adjustment
void ChannelCaHVACompCategory::computeTadj(RNG& rng)
{
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
void ChannelCaHVACompCategory::count()
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
    printf("Total CaHVA Channel = %lld, Mean = %lf, StDev = %lf\n", totalCount,
           mean, std);
}
