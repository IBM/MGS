#include "Lens.h"
#include "AMPAReceptor_MarkovCompCategory.h"
#include "NDPairList.h"
#include "CG_AMPAReceptor_MarkovCompCategory.h"
#include <mpi.h>

AMPAReceptor_MarkovCompCategory::AMPAReceptor_MarkovCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
   : CG_AMPAReceptor_MarkovCompCategory(sim, modelName, ndpList)
{
}

// GOAL:
//  1. find Q10 adjustment
void AMPAReceptor_MarkovCompCategory::computeTadj(RNG& rng)
{
  // Step 1. Find temperature adjustment factor Tadj
  //      based upon Q10 and T values
  assert(*(getSharedMembers().T) > 273.15);
  // if (getSharedMembers().T and getSharedMembers().Tadj)
  if (getSharedMembers().T)
    getSharedMembers().Tadj = pow(
        Q10, ((*(getSharedMembers().T) - 273.15 - BASED_TEMPERATURE) / 10.0));
  // pow(static_cast<dyn_var_t>(Q10), ((*(getSharedMembers().T) - 273.15 -
  // BASED_TEMPERATURE) / 10.0));
  //(((*(getSharedMembers().T) - 273.15 - BASED_TEMPERATURE) / 10.0));
}

// Return the statistics of distributing the instances of this nodetype
// onto different computing nodes
// i.e. the total instances,
//      the mean #-of-instance-being-processed by each node,
//      the stddev #-of-instance-being-processed by each node
void AMPAReceptor_MarkovCompCategory::count()
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
    printf("Total AMPA synapse = %lld, Mean = %lf, StDev = %lf\n", totalCount,
           mean, std);
}
