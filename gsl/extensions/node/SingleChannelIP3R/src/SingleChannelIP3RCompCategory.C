// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#include "Mgs.h"
#include "SingleChannelIP3RCompCategory.h"
#include "NDPairList.h"
#include "CG_SingleChannelIP3RCompCategory.h"
#include <mpi.h>
#include <math.h>
#include "NumberUtils.h"

#include "Params.h"

SingleChannelIP3RCompCategory::SingleChannelIP3RCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
   : CG_SingleChannelIP3RCompCategory(sim, modelName, ndpList)
{
}

void SingleChannelIP3RCompCategory::computeTadj(RNG& rng) 
{
  // Step 1. Find temperature adjustment factor Tadj
  //      based upon Q10 and T values
  // if (getSharedMembers().T and getSharedMembers().Tadj)
  if (getSharedMembers().T)
  {
    assert(*(getSharedMembers().T) > 273.15);
    getSharedMembers().Tadj = pow(
        Q10, ((*(getSharedMembers().T) - 273.15 - BASED_TEMPERATURE) / 10.0));
  }
}

void SingleChannelIP3RCompCategory::setupChannel(RNG& rng)
{
  /*
read_LCC_Markov(LCC_Markov_filename, lcc_matChannelRateConstant, mL,
vOpenStates, lcc_initialstate)
OR
ALLOCATE(lcc_Ktransitionrate(mL,mL))
read_LCC_Markov_Sun2000(filename, akL, mL, vOpenStates, initial_state, P_dhprT)
   */

  Params param;
  param.readMarkovModel(getSharedMembers().SingleChannelModelFileName.c_str(),
                        getSharedMembers().matChannelRateConstant,
                        getSharedMembers().numChanStates,
                        getSharedMembers().vOpenStates,
                        getSharedMembers().initialstate);
}



SingleChannelIP3RCompCategory::~SingleChannelIP3RCompCategory()
{
  /*for (int ii=0; ii < numStates; ii++)
  {
          delete matChannelRateConstant[ii];
  }
  delete []matChannelRateConstant;
  delete vOpenStates;
  */
  delete[] getSharedMembers().matChannelRateConstant;
  delete getSharedMembers().vOpenStates;
}


//
// Return the statistics of distributing the instances of this nodetype
// onto different computing nodes
// i.e. the total instances,
//      the mean #-of-instance-being-processed by each node,
//      the stddev #-of-instance-being-processed by each node
void SingleChannelIP3RCompCategory::count()
{
  long long totalCount, localCount = _nodes.size();
  MPI_Allreduce((void*)&localCount, (void*)&totalCount, 1, MPI_LONG_LONG,
                MPI_SUM, MPI_COMM_WORLD);
  float mean = (float(totalCount)) / getSimulation().getNumProcesses();
  float localVariance = Square(float(localCount) - mean);
  float totalVariance;
  MPI_Allreduce((void*)&localVariance, (void*)&totalVariance, 1, MPI_FLOAT,
                MPI_SUM, MPI_COMM_WORLD);
  float std = sqrt(totalVariance / getSimulation().getNumProcesses());

  if (getSimulation().getRank() == 0)
    printf("Total cpts having IP3R Channel = %lld, Mean = %lf, StDev = %lf\n",
           totalCount, mean, std);
}
