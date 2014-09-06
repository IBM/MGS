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
#include "ChannelHodgkinNatCompCategory.h"
#include "NDPairList.h"
#include "CG_ChannelHodgkinNatCompCategory.h"
#include <math.h>
//#define DEBUG_HH

ChannelHodgkinNatCompCategory::ChannelHodgkinNatCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
   : CG_ChannelHodgkinNatCompCategory(sim, modelName, ndpList)
{
}

void ChannelHodgkinNatCompCategory::computeE(RNG& rng) 
{
  if (getSharedMembers().T && getSharedMembers().Na_EC && getSharedMembers().Na_IC) {
    getSharedMembers().E_Na.push_back(0.08686 * *(getSharedMembers().T) * log(*(getSharedMembers().Na_EC) / *(getSharedMembers().Na_IC)));
#ifdef DEBUG_HH  
    std::cerr<<getSimulation().getRank()<<" : T="<<*getSharedMembers().T<<" Na_EC="<<*getSharedMembers().Na_EC<<" Na_IC=";
    std::cerr<<*getSharedMembers().Na_IC<<" E_Na="<<getSharedMembers().E_Na[0]<<std::endl;
#endif
  }
}

void ChannelHodgkinNatCompCategory::count() 
{
  long long totalCount, localCount=_nodes.size();
  MPI_Allreduce((void*) &localCount, (void*) &totalCount, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  float localVar, totalVar, mean=float(totalCount)/getSimulation().getNumProcesses();
  localVar=(float(localCount)-mean)*(float(localCount)-mean);
  MPI_Allreduce((void*) &localVar, (void*) &totalVar, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  float std=sqrt(totalVar/getSimulation().getNumProcesses());
  if (getSimulation().getRank()==0) printf("Total Nat Channel = %lld, Mean = %lf, StDev = %lf\n", totalCount, mean, std);
}
