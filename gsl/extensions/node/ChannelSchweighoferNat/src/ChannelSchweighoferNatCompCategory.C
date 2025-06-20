// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Mgs.h"
#include "ChannelSchweighoferNatCompCategory.h"
#include "NDPairList.h"
#include "CG_ChannelSchweighoferNatCompCategory.h"
#include <math.h>
//#define DEBUG_HH

ChannelSchweighoferNatCompCategory::ChannelSchweighoferNatCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
   : CG_ChannelSchweighoferNatCompCategory(sim, modelName, ndpList)
{
}

void ChannelSchweighoferNatCompCategory::computeE(RNG& rng) 
{
  if (getSharedMembers().T && getSharedMembers().Na_EC && getSharedMembers().Na_IC) {
    getSharedMembers().E_Na.push_back(0.08686 * *(getSharedMembers().T) * log(*(getSharedMembers().Na_EC) / *(getSharedMembers().Na_IC)));
#ifdef DEBUG_HH  
    std::cerr<<getSimulation().getRank()<<" : T="<<*getSharedMembers().T<<" Na_EC="<<*getSharedMembers().Na_EC<<" Na_IC=";
    std::cerr<<*getSharedMembers().Na_IC<<" E_Na="<<getSharedMembers().E_Na[0]<<std::endl;
#endif
  }
}

void ChannelSchweighoferNatCompCategory::count() 
{
  long long totalCount, localCount=_nodes.size();
  MPI_Allreduce((void*) &localCount, (void*) &totalCount, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  float localVar, totalVar, mean=float(totalCount)/getSimulation().getNumProcesses();
  localVar=(float(localCount)-mean)*(float(localCount)-mean);
  MPI_Allreduce((void*) &localVar, (void*) &totalVar, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  float std=sqrt(totalVar/getSimulation().getNumProcesses());
  if (getSimulation().getRank()==0) printf("Total Nat Channel = %lld, Mean = %lf, StDev = %lf\n", totalCount, mean, std);
}
