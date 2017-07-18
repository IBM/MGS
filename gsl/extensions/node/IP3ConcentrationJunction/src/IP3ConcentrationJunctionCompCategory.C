// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "Lens.h"
#include "IP3ConcentrationJunctionCompCategory.h"
#include "NDPairList.h"
#include "CG_IP3ConcentrationJunctionCompCategory.h"
#include <math.h>
//#define DEBUG_HH

IP3ConcentrationJunctionCompCategory::IP3ConcentrationJunctionCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
   : CG_IP3ConcentrationJunctionCompCategory(sim, modelName, ndpList)
{
}

void IP3ConcentrationJunctionCompCategory::deriveParameters(RNG& rng) 
{
  if (getSharedMembers().deltaT) {
    getSharedMembers().bmt = 2.0 / (getSharedMembers().beta * *(getSharedMembers().deltaT)) ;
#ifdef DEBUG_HH
    std::cerr<<getSimulation().getRank()<<" : IP3ConcentrationJunctions : "<<_nodes.size()<<std::endl;
#endif
  }
}

void IP3ConcentrationJunctionCompCategory::count() 
{
  long long totalCount, localCount=_nodes.size();
  MPI_Allreduce((void*) &localCount, (void*) &totalCount, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  float localVar, totalVar, mean=float(totalCount)/getSimulation().getNumProcesses();
  localVar=(float(localCount)-mean)*(float(localCount)-mean);
  MPI_Allreduce((void*) &localVar, (void*) &totalVar, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  float std=sqrt(totalVar/getSimulation().getNumProcesses());
  if (getSimulation().getRank()==0) printf("Total Calcium Junction = %lld, Mean = %lf, StDev = %lf\n", totalCount, mean, std);
}
