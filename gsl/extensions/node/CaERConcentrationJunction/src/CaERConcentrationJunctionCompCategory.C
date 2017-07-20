#include "Lens.h"
#include "CaERConcentrationJunctionCompCategory.h"
#include "NDPairList.h"
#include "CG_CaERConcentrationJunctionCompCategory.h"
#include <mpi.h>

CaERConcentrationJunctionCompCategory::CaERConcentrationJunctionCompCategory(
    Simulation& sim, const std::string& modelName, const NDPairList& ndpList)
    : CG_CaERConcentrationJunctionCompCategory(sim, modelName, ndpList)
{
}

void CaERConcentrationJunctionCompCategory::deriveParameters(RNG& rng)
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
    std::cerr << getSimulation().getRank()
              << " : CaERConcentrationJunctions : " << _nodes.size()
              << std::endl;
#endif
  }
}

void CaERConcentrationJunctionCompCategory::count()
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
    printf("Total CaER Junction = %lld, Mean = %lf, StDev = %lf\n",
           totalCount, mean, std);
}
