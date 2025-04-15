// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Lens.h"
#include "MihalasNieburIAFUnitCompCategory.h"
#include "NDPairList.h"
#include "CG_MihalasNieburIAFUnitCompCategory.h"
#include <fstream>
#include <sstream>

#define SHD getSharedMembers()
#define ITER getSimulation().getIteration()

MihalasNieburIAFUnitCompCategory::MihalasNieburIAFUnitCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
   : CG_MihalasNieburIAFUnitCompCategory(sim, modelName, ndpList)
{
}

void MihalasNieburIAFUnitCompCategory::initializeShared(RNG& rng) 
{
  SHD.G=SHD.GoC*SHD.C;
  int nI=SHD.k.size();
  assert(SHD.R.size()==nI);
  assert(SHD.A.size()==nI);
  
  int n=SHD.collectWeightsOn.size();
  if (n>0) {
    for (int i=0; i<n; ++i) {
      std::ofstream fs; std::ostringstream os;
      os<<SHD.weightsFileName<<"_"<<SHD.collectWeightsOn[i]<<".dat";
      fs.open(os.str().c_str(), std::ofstream::trunc|std::ofstream::out);
      fs.close();
    }
  }
  
}

void MihalasNieburIAFUnitCompCategory::outputWeightsShared(RNG& rng) 
{
  if (SHD.collectWeightsOn.size()>0) {
    if (SHD.collectWeightsOn[SHD.collectWeightsNext]==ITER) {
      if (SHD.collectWeightsOn.size()-1>SHD.collectWeightsNext) ++SHD.collectWeightsNext;
      int rank=getSimulation().getRank();
      int n=0;
      while (n<getSimulation().getNumProcesses()) {
        if (n==rank) {
          std::ofstream fs; 
          std::ostringstream os;
          os<<SHD.weightsFileName<<"_"<<ITER<<".dat";
          fs.open(os.str().c_str(), std::ofstream::app|std::ofstream::out);
          ShallowArray<MihalasNieburIAFUnit>::iterator it = _nodes.begin();
          ShallowArray<MihalasNieburIAFUnit>::iterator end = _nodes.end();
          for (; it != end; ++it) (*it).outputWeights(fs);
          fs.close();
        }
        ++n;
        MPI_Barrier(MPI_COMM_WORLD);
      }
    }
  }
}

