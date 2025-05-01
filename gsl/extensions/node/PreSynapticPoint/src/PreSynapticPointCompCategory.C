// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Mgs.h"
#include "PreSynapticPointCompCategory.h"
#include "NDPairList.h"
#include "CG_PreSynapticPointCompCategory.h"
#include <mpi.h>

PreSynapticPointCompCategory::PreSynapticPointCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
   : CG_PreSynapticPointCompCategory(sim, modelName, ndpList)
{
}

void PreSynapticPointCompCategory::count()
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
    printf("Total PreSynapticPoint = %lld, Mean = %lf, StDev = %lf\n",
           totalCount, mean, std);
}
