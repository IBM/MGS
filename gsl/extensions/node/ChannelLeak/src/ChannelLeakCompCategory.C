/* =================================================================
Licensed Materials - Property of IBM

"Restricted Materials of IBM"

BMC-YKT-07-18-2017

(C) Copyright IBM Corp. 2005-2017  All rights reserved

US Government Users Restricted Rights -
Use, duplication or disclosure restricted by
GSA ADP Schedule Contract with IBM Corp.

=================================================================
*/


#include "Lens.h"
#include "ChannelLeakCompCategory.h"
#include "NDPairList.h"
#include "CG_ChannelLeakCompCategory.h"
#include <math.h>

#include <mpi.h>
#include "NumberUtils.h"


ChannelLeakCompCategory::ChannelLeakCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
   : CG_ChannelLeakCompCategory(sim, modelName, ndpList)
{
}
void ChannelLeakCompCategory::count()
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
    printf("Total Leak Channel = %lld, Mean = %lf, StDev = %lf\n", totalCount,
           mean, std);
}


