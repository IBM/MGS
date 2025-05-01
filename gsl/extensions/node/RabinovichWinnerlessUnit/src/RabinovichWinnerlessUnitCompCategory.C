// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Mgs.h"
#include "RabinovichWinnerlessUnitCompCategory.h"
#include "NDPairList.h"
#include "CG_RabinovichWinnerlessUnitCompCategory.h"
#include <fstream>
#include <sstream>

#define SHD getSharedMembers()
#define ITER getSimulation().getIteration()

RabinovichWinnerlessUnitCompCategory::RabinovichWinnerlessUnitCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
   : CG_RabinovichWinnerlessUnitCompCategory(sim, modelName, ndpList)
{
}

void RabinovichWinnerlessUnitCompCategory::initializeShared(RNG& rng) 
{
  // Setup FN steps
  SHD.step1=SHD.deltaT/SHD.tau1;
  SHD.step2=SHD.deltaT;
  SHD.step3=SHD.deltaT/SHD.tau2;

  // Setup when weights are saved
  std::ostringstream sysCall;
  sysCall<<"mkdir -p "<<SHD.sharedDirectory.c_str()<<";";
  try {
    int systemRet = system(sysCall.str().c_str());
    if (systemRet == -1)
      throw;
  } catch(...) {};
  
  int n=SHD.collectWeightsOn.size();
  if (n>0) {
    for (int i=0; i<n; ++i) {
      std::ofstream fs; std::ostringstream os;
      os<<SHD.sharedDirectory.c_str()<<"MSN_LN_weights"
        <<"_"<<SHD.collectWeightsOn[i]<<SHD.sharedFileExt.c_str();
      fs.open(os.str().c_str(), std::ofstream::trunc|std::ofstream::out);
      fs.close();
      os.str("");
      os<<SHD.sharedDirectory.c_str()<<"Cx2Str_weights"
        <<"_"<<SHD.collectWeightsOn[i]<<SHD.sharedFileExt.c_str();
      fs.open(os.str().c_str(), std::ofstream::trunc|std::ofstream::out);
      fs.close();
      
      os.str("");
      os<<SHD.sharedDirectory.c_str()<<"DA2Str_weights"
        <<"_"<<SHD.collectWeightsOn[i]<<SHD.sharedFileExt.c_str();
      fs.open(os.str().c_str(), std::ofstream::trunc|std::ofstream::out);
      fs.close();
    }
  }
}

void RabinovichWinnerlessUnitCompCategory::outputWeightsShared(RNG& rng) 
{
  if (SHD.collectWeightsOn.size()>0) {
    if (SHD.collectWeightsOn[SHD.collectWeightsNext]==ITER) {
      if (SHD.collectWeightsOn.size()-1 > SHD.collectWeightsNext)
        ++SHD.collectWeightsNext;
      int rank=getSimulation().getRank();
      int n=0;
      while (n<getSimulation().getNumProcesses()) {
        if (n==rank) {
          std::ofstream fsLN, fsDR, fsNS; 
          std::ostringstream os;
          os<<SHD.sharedDirectory.c_str()<<"MSN_LN_weights"
            <<"_"<<ITER<<SHD.sharedFileExt.c_str();
	  fsLN.open(os.str().c_str(), std::ofstream::app|std::ofstream::out);
 
	  os.str("");
          os<<SHD.sharedDirectory.c_str()<<"Cx2Str_weights"
            <<"_"<<ITER<<SHD.sharedFileExt.c_str();
	  fsDR.open(os.str().c_str(), std::ofstream::app|std::ofstream::out);

          os.str("");
          os<<SHD.sharedDirectory.c_str()<<"DA2Str_weights"
            <<"_"<<ITER<<SHD.sharedFileExt.c_str();
	  fsNS.open(os.str().c_str(), std::ofstream::app|std::ofstream::out);

	  ShallowArray<RabinovichWinnerlessUnit>::iterator it = _nodes.begin();
          ShallowArray<RabinovichWinnerlessUnit>::iterator end = _nodes.end();
          for (; it != end; ++it)
            (*it).outputWeights(fsLN, fsDR, fsNS);
          fsLN.close();
          fsDR.close();
    	  fsNS.close();
        }
        ++n;
        MPI_Barrier(MPI_COMM_WORLD);
      }
    }
  }
}

