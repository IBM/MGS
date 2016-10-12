#include "Lens.h"
#include "mGluReceptorType1CompCategory.h"
#include "NDPairList.h"
#include "CG_mGluReceptorType1CompCategory.h"

#include <math.h>
#include <mpi.h>

mGluReceptorType1CompCategory::mGluReceptorType1CompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
   : CG_mGluReceptorType1CompCategory(sim, modelName, ndpList)
{
}

//void mGluReceptorType1CompCategory::computeTadj(RNG& rng) 
//{
//}

// Return the statistics of distributing the instances of this nodetype
// onto different computing nodes
// i.e. the total instances,
//      the mean #-of-instance-being-processed by each node,
//      the stddev #-of-instance-being-processed by each node
void mGluReceptorType1CompCategory::count()
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
    printf("Total mGluR-I synapse = %lld, Mean = %lf, StDev = %lf\n", totalCount,
           mean, std);
}
