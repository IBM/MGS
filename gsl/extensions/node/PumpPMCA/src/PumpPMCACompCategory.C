#include "Lens.h"
#include "PumpPMCACompCategory.h"
#include "NDPairList.h"
#include "CG_PumpPMCACompCategory.h"

#include "NumberUtils.h"

PumpPMCACompCategory::PumpPMCACompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
   : CG_PumpPMCACompCategory(sim, modelName, ndpList)
{
}

// GOAL:
//  1. find Q10 adjustment
void PumpPMCACompCategory::computeTadj(RNG& rng) 
{
  // Step 1. Find temperature adjustment factor Tadj
  //      based upon Q10 and T values
  assert((getSharedMembers().T));
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
void PumpPMCACompCategory::count()
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
    printf("Total PMCA Channel = %lld, Mean = %lf, StDev = %lf\n", totalCount,
           mean, std);
}


