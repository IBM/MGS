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
#include "CaConcentrationJunctionCompCategory.h"
#include "NDPairList.h"
#include "CG_CaConcentrationJunctionCompCategory.h"
#include <math.h>
//#define DEBUG_HH

CaConcentrationJunctionCompCategory::CaConcentrationJunctionCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
   : CG_CaConcentrationJunctionCompCategory(sim, modelName, ndpList)
{
}

void CaConcentrationJunctionCompCategory::deriveParameters(RNG& rng) 
{
  if (getSharedMembers().deltaT) {
#if CALCIUM_CYTO_DYNAMICS == FAST_BUFFERING
    getSharedMembers().bmt = 2.0 / (getSharedMembers().beta * *(getSharedMembers().deltaT)) ;
#elif CALCIUM_CYTO_DYNAMICS ==  REGULAR_DYNAMICS
    getSharedMembers().bmt =
        2.0 / ( *(getSharedMembers().deltaT));
#else
    assert(0);
#endif
    assert(getSharedMembers().bmt > 0);
    getSharedMembers().x_bmt =
        2.0 / ( *(getSharedMembers().deltaT));

#ifdef DEBUG_HH
    std::cerr<<getSimulation().getRank()<<" : CaConcentrationJunctions : "<<_nodes.size()<<std::endl;
#endif
  }
}

void CaConcentrationJunctionCompCategory::count() 
{
  long long totalCount, localCount=_nodes.size();
  MPI_Allreduce((void*) &localCount, (void*) &totalCount, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  float localVar, totalVar, mean=float(totalCount)/getSimulation().getNumProcesses();
  localVar=(float(localCount)-mean)*(float(localCount)-mean);
  MPI_Allreduce((void*) &localVar, (void*) &totalVar, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  float std=sqrt(totalVar/getSimulation().getNumProcesses());
  if (getSimulation().getRank()==0) printf("Total Calcium Junction = %lld, Mean = %lf, StDev = %lf\n", totalCount, mean, std);
}
