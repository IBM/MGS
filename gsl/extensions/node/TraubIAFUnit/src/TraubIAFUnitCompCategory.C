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
#include "TraubIAFUnitCompCategory.h"
#include "NDPairList.h"
#include "CG_TraubIAFUnitCompCategory.h"
#include <fstream>
#include <sstream>

#define SHD getSharedMembers()
#define ITER getSimulation().getIteration()

TraubIAFUnitCompCategory::TraubIAFUnitCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
   : CG_TraubIAFUnitCompCategory(sim, modelName, ndpList)
{
}

void TraubIAFUnitCompCategory::initializeShared(RNG& rng) 
{
  std::ostringstream sysCall;
  sysCall<<"mkdir -p "<<SHD.sharedDirectory.c_str()<<";";
  try {
    int systemRet = system(sysCall.str().c_str());
    if (systemRet == -1)
      throw;
  } catch(...) {};
  int rank=getSimulation().getRank();
  int n=0;  
  if (SHD.op_saveWeights)
    {  
      // Take it in turn opening and creating the file to create the stream on each node
      while (n<getSimulation().getNumProcesses()) {
        if (n==rank) {  
          os_weight<<SHD.sharedDirectory<<"Weights"<<SHD.sharedFileExt;
          weight_file=new std::ofstream(os_weight.str().c_str(),
                                        std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
          weight_file->close();
        }
        ++n;
        MPI_Barrier(MPI_COMM_WORLD); // wait node creating the stream to finish
      }
      // Now take it in turn writing to the file, where rank 0 also clears the file.
      n=0;
      while (n<getSimulation().getNumProcesses()) {
        if (n==rank) {
          ShallowArray<TraubIAFUnit>::iterator it = _nodes.begin();
          ShallowArray<TraubIAFUnit>::iterator end = _nodes.end();
          weight_file->open(os_weight.str().c_str(),
                            std::ofstream::out | std::ofstream::app | std::ofstream::binary);
          for (; it != end; ++it)
            (*it).outputWeights(*weight_file);
          weight_file->close();
        }
        ++n;    
        MPI_Barrier(MPI_COMM_WORLD); // wait for node writing to finish
      }
    }
  if (SHD.op_saveGJs)
    {
      n=0;
      // Take it in turn opening and creating the file to create the stream on each node
      while (n<getSimulation().getNumProcesses()) {
        if (n==rank) {  
          os_GJ<<SHD.sharedDirectory<<"GJs"<<SHD.sharedFileExt;
          GJ_file=new std::ofstream(os_GJ.str().c_str(),
                                        std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
          GJ_file->close();
        }
        ++n;
        MPI_Barrier(MPI_COMM_WORLD); // wait node creating the stream to finish
      }
      // Now take it in turn writing to the file, where rank 0 also clears the file.
      n=0;
      while (n<getSimulation().getNumProcesses()) {
        if (n==rank) {
          ShallowArray<TraubIAFUnit>::iterator it = _nodes.begin();
          ShallowArray<TraubIAFUnit>::iterator end = _nodes.end();
          GJ_file->open(os_GJ.str().c_str(),
                            std::ofstream::out | std::ofstream::app | std::ofstream::binary);
          for (; it != end; ++it)
            (*it).outputGJs(*GJ_file);
          GJ_file->close();
        }
        ++n;    
        MPI_Barrier(MPI_COMM_WORLD); // wait for node writing to finish
      }
    }
  if (SHD.op_savePSPs)
    {
      // If saving PSPs, take it in turn to create the stream
      n=0;
      while (n<getSimulation().getNumProcesses()) {
        if (n==rank) {  
          os_psp<<SHD.sharedDirectory<<"PSPs"<<SHD.sharedFileExt;
          psp_file=new std::ofstream(os_psp.str().c_str(),
                                     std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
          psp_file->close();
        }
        ++n;
        MPI_Barrier(MPI_COMM_WORLD); // wait for node creating the stream to finish
      }
    }
}

void TraubIAFUnitCompCategory::outputPSPsShared(RNG& rng) 
{
  if (SHD.op_savePSPs)
    {
      int rank=getSimulation().getRank();
      int n=0;  
      while (n<getSimulation().getNumProcesses()) {
        if (n==rank) {
          ShallowArray<TraubIAFUnit>::iterator it = _nodes.begin();
          ShallowArray<TraubIAFUnit>::iterator end = _nodes.end();
          psp_file->open(os_psp.str().c_str(),
                         std::ofstream::out | std::ofstream::app | std::ofstream::binary);
          for (; it != end; ++it)
            (*it).outputPSPs(*psp_file);
          psp_file->close();
        }
        ++n;    
        MPI_Barrier(MPI_COMM_WORLD);
      }
    }
}

