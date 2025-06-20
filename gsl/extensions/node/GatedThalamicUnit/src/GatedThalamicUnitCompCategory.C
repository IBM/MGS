// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#include "Mgs.h"
#include "GatedThalamicUnitCompCategory.h"
#include "NDPairList.h"
#include "CG_GatedThalamicUnitCompCategory.h"
#include <fstream>
#include <sstream>
#include <iostream>

#define SHD getSharedMembers()
#define ITER getSimulation().getIteration()

GatedThalamicUnitCompCategory::GatedThalamicUnitCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
   : CG_GatedThalamicUnitCompCategory(sim, modelName, ndpList)
{
}

void GatedThalamicUnitCompCategory::initializeShared(RNG& rng) 
{
  getSharedMembers().alphaZ=1.0/double(getSharedMembers().tauZ);
  int n=SHD.collectWeightsOn.size();
  if (n>0) {
    std::ofstream fs; std::ostringstream os;
    for (int i=0; i<n; ++i) {
      os.str("");
      os<<"PH_"<<SHD.weightsFileName<<"_"<<SHD.collectWeightsOn[i]<<".dat";
      fs.open(os.str().c_str(), std::ofstream::trunc|std::ofstream::out);
      fs.close();
    }
  }
}

void GatedThalamicUnitCompCategory::outputWeightsShared(RNG& rng) 
{
  if (SHD.collectWeightsOn.size()>0) {
    if (SHD.collectWeightsOn[SHD.collectWeightsNext]==ITER) {
      if (SHD.collectWeightsOn.size()-1>SHD.collectWeightsNext) ++SHD.collectWeightsNext;
      int rank=getSimulation().getRank();
      int n=0;
      while (n<getSimulation().getNumProcesses()) {
	if (n==rank) {
	  std::ofstream fsPH;
	  std::ostringstream os;
	  os<<"PH_"<<SHD.weightsFileName<<"_"<<ITER<<".dat";
	  fsPH.open(os.str().c_str(), std::ofstream::app|std::ofstream::out);
	  fsPH.precision(8);
	  ShallowArray<GatedThalamicUnit>::iterator it = _nodes.begin();
	  ShallowArray<GatedThalamicUnit>::iterator end = _nodes.end();
	  for (; it != end; ++it) (*it).outputWeights(fsPH);
	  fsPH.close();
	}
	++n;
	MPI_Barrier(MPI_COMM_WORLD);
      }
    }
  }
}

void GatedThalamicUnitCompCategory::inputWeightsShared(RNG& rng) 
{
  if (SHD.loadWeightsOn.size()>0) {
    if (SHD.loadWeightsOn[SHD.loadWeightsNext]==ITER) {
      if (SHD.loadWeightsOn.size()-1>SHD.loadWeightsNext) ++SHD.loadWeightsNext;
      int rank=getSimulation().getRank();
      int n=0;
      while (n<getSimulation().getNumProcesses()) {
	if (n==rank) {
	  std::ifstream fsPH;
	  std::ostringstream os;
	  os<<"PH_"<<SHD.weightsFileName<<"_"<<ITER<<".dat";
	  fsPH.open(os.str().c_str(), std::ofstream::app|std::ofstream::out);
	  if (fsPH.good()) {
	    fsPH.precision(8);
	    int row, col;
	    ShallowArray<GatedThalamicUnit>::iterator it, end;
	    while (!fsPH.eof()) {
	      fsPH>>row>>col;
	      ShallowArray<GatedThalamicUnit>::iterator end = _nodes.end();
	      for (it = _nodes.begin(); it != end; ++it) {
		if (it->getGlobalIndex()+1==row) {
		  it->inputWeight(fsPH, col);
		  break;
		}
	      }
	      if (it==end) {
		double nextWeight;
		fsPH>>nextWeight;
	      }
	    }
	    fsPH.close();
	  }
	}
	++n;
	MPI_Barrier(MPI_COMM_WORLD);
      }
    }
  }
}
