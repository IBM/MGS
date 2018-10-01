// =================================================================
//
// (C) Copyright 2018 New Jersey Institute of Technology.
//
// =================================================================


#include "Lens.h"
#include "ChannelLeak_STR_FSI_mouseCompCategory.h"
#include "NDPairList.h"
#include "CG_ChannelLeak_STR_FSI_mouseCompCategory.h"
#include <math.h>

#include <mpi.h>
#include "NumberUtils.h"


ChannelLeak_STR_FSI_mouseCompCategory::ChannelLeak_STR_FSI_mouseCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
   : CG_ChannelLeak_STR_FSI_mouseCompCategory(sim, modelName, ndpList)
{
}
void ChannelLeak_STR_FSI_mouseCompCategory::count()
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
    printf("Total FSI Leak Channel = %lld, Mean = %lf, StDev = %lf\n", totalCount,
           mean, std);
}


