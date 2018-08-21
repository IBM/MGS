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
#include "ChannelHayKpCompCategory.h"
#include "NDPairList.h"
#include "CG_ChannelHayKpCompCategory.h"
#include <math.h>
//#define DEBUG_HH

ChannelHayKpCompCategory::ChannelHayKpCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
   : CG_ChannelHayKpCompCategory(sim, modelName, ndpList)
{
}

void ChannelHayKpCompCategory::computeE(RNG& rng) 
{
  if (getSharedMembers().T && getSharedMembers().K_EC && getSharedMembers().K_IC) {
    getSharedMembers().E_K.push_back(0.08686 * *(getSharedMembers().T) * log(*(getSharedMembers().K_EC) / *(getSharedMembers().K_IC)));
#ifdef DEBUG_HH  
    std::cerr<<getSimulation().getRank()<<" : T="<<*getSharedMembers().T<<" K_EC="<<*getSharedMembers().K_EC<<" K_IC=";
    std::cerr<<*getSharedMembers().K_IC<<" E_K="<<getSharedMembers().E_K[0]<<std::endl;
#endif
  }
}

void ChannelHayKpCompCategory::count() 
{
  long long totalCount, localCount=_nodes.size();
  MPI_Allreduce((void*) &localCount, (void*) &totalCount, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  float localVar, totalVar, mean=float(totalCount)/getSimulation().getNumProcesses();
  localVar=(float(localCount)-mean)*(float(localCount)-mean);
  MPI_Allreduce((void*) &localVar, (void*) &totalVar, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  float std=sqrt(totalVar/getSimulation().getNumProcesses());
  if (getSimulation().getRank()==0) printf("Total Kp Channel = %lld, Mean = %lf, StDev = %lf\n", totalCount, mean, std);
}
