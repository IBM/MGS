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

