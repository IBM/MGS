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
#include "SpineIAFUnitCompCategory.h"
#include "NDPairList.h"
#include "CG_SpineIAFUnitCompCategory.h"
#include <fstream>
#include <sstream>

#define SHD getSharedMembers()
#define ITER getSimulation().getIteration()

SpineIAFUnitCompCategory::SpineIAFUnitCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList)
  : CG_SpineIAFUnitCompCategory(sim, modelName, ndpList)
{
  SHD.collectWeightsNext = 0;
}

void SpineIAFUnitCompCategory::initializeShared(RNG& rng)
{
  std::ostringstream sysCall;
  sysCall<<"mkdir -p "<<SHD.sharedDirectory.c_str()<<";";
  try {
    int systemRet = system(sysCall.str().c_str());
    if (systemRet == -1)
      throw;
  } catch(...) {};  
  if (SHD.op_saveWeights)
    {
      int n=SHD.collectWeightsOn.size();
      if (n>0)
        {
          int rank=getSimulation().getRank();          
          for (int i=0; i<n; i++)
            {
              int r=0;
              while (r<getSimulation().getNumProcesses())
                {                  
                  if (r==rank)
                    {
                      os_weights.str(std::string());
                      os_weights<<SHD.sharedDirectory<<"AMPAWeights_"<<SHD.collectWeightsOn[i]
                                <<SHD.sharedFileExt;
                      weights_file=new std::ofstream(os_weights.str().c_str(),
                                                     std::ofstream::out | std::ofstream::trunc
                                                     | std::ofstream::binary);
                      weights_file->close();
                    }                  
                  ++r;
                  MPI_Barrier(MPI_COMM_WORLD); // wait node creating the stream to finish
                }
            }
        }
    }
}

void SpineIAFUnitCompCategory::outputWeightsShared(RNG& rng)
{
  if (SHD.op_saveWeights)
    {
      int n=SHD.collectWeightsOn.size();
      if (SHD.collectWeightsOn[SHD.collectWeightsNext]==ITER)
        {
          os_weights.str(std::string());
          os_weights<<SHD.sharedDirectory<<"AMPAWeights_"
                    <<SHD.collectWeightsOn[SHD.collectWeightsNext]
                    <<SHD.sharedFileExt;
          if (SHD.collectWeightsOn.size()-1 > SHD.collectWeightsNext)
            SHD.collectWeightsNext++;
          int rank=getSimulation().getRank();
          int r=0;
          while (r<getSimulation().getNumProcesses())
            {
              if (r==rank) {
                ShallowArray<SpineIAFUnit>::iterator it = _nodes.begin();
                ShallowArray<SpineIAFUnit>::iterator end = _nodes.end();
                weights_file->open(os_weights.str().c_str(),
                                   std::ofstream::out | std::ofstream::app | std::ofstream::binary);
                for (; it != end; ++it)
                  (*it).outputWeights(*weights_file);
                weights_file->close();
              }
              ++r;
              MPI_Barrier(MPI_COMM_WORLD); // wait for node writing to finish
            }
        }
    }
}

