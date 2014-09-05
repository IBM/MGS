// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "Lens.h"
#include "NMDAReceptorCompCategory.h"
#include "NDPairList.h"
#include "CG_NMDAReceptorCompCategory.h"
#include <math.h>

NMDAReceptorCompCategory::NMDAReceptorCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
   : CG_NMDAReceptorCompCategory(sim, modelName, ndpList)
{
}

void NMDAReceptorCompCategory::count() 
{
  long long totalCount, localCount=_nodes.size();
  MPI_Allreduce((void*) &localCount, (void*) &totalCount, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  float localVar, totalVar, mean=float(totalCount)/getSimulation().getNumProcesses();
  localVar=(float(localCount)-mean)*(float(localCount)-mean);
  MPI_Allreduce((void*) &localVar, (void*) &totalVar, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  float std=sqrt(totalVar/getSimulation().getNumProcesses());
  if (getSimulation().getRank()==0) printf("Total NMDA synapse = %lld, Mean = %lf, StDev = %lf\n", totalCount, mean, std);
}
