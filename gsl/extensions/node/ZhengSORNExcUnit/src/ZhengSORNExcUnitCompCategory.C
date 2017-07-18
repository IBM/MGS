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
#include "ZhengSORNExcUnitCompCategory.h"
#include "NDPairList.h"
#include "CG_ZhengSORNExcUnitCompCategory.h"
#include <fstream>
#include <sstream>
#include <iostream>

#define SHD getSharedMembers()
#define ITER getSimulation().getIteration()

ZhengSORNExcUnitCompCategory::ZhengSORNExcUnitCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
   : CG_ZhengSORNExcUnitCompCategory(sim, modelName, ndpList)
{
}

void ZhengSORNExcUnitCompCategory::initializeShared(RNG& rng) 
{
  SHD.eta_iSTDP = (1.0+1.0/SHD.mu_iSTDP);
  SHD.eta_iLTP = SHD.eta_inhib * (1.0-SHD.eta_iSTDP);
  int n=SHD.collectWeightsOn.size();
  if (n>0) {
    for (int i=0; i<n; ++i) {
      std::ofstream fsE2E, fsI2E; std::ostringstream os;
      os.str("");
      os<<"E2E_"<<SHD.weightsFileName<<"_"<<SHD.collectWeightsOn[i]<<".dat";
      fsE2E.open(os.str().c_str(), std::ofstream::trunc|std::ofstream::out);
      fsE2E.close();
      os.str(""); 
      os<<"I2E_"<<SHD.weightsFileName<<"_"<<SHD.collectWeightsOn[i]<<".dat";
      fsI2E.open(os.str().c_str(), std::ofstream::trunc|std::ofstream::out);
      fsI2E.close();
    }
  }
}

void ZhengSORNExcUnitCompCategory::outputWeightsShared(RNG& rng) 
{
  if (SHD.collectWeightsOn.size()>0) {
    if (SHD.collectWeightsOn[SHD.collectWeightsNext]==ITER) {
      if (SHD.collectWeightsOn.size()-1>SHD.collectWeightsNext) ++SHD.collectWeightsNext;
      int rank=getSimulation().getRank();
      int n=0;
      while (n<getSimulation().getNumProcesses()) {
	if (n==rank) {
	  std::ofstream fsE2E, fsI2E; 
	  std::ostringstream os;
	  os<<"E2E_"<<SHD.weightsFileName<<"_"<<ITER<<".dat";
	  fsE2E.open(os.str().c_str(), std::ofstream::app|std::ofstream::out);
	  os.str("");
	  os<<"I2E_"<<SHD.weightsFileName<<"_"<<ITER<<".dat";
	  fsI2E.open(os.str().c_str(), std::ofstream::app|std::ofstream::out);
	  
          ShallowArray<ZhengSORNExcUnit>::iterator it = _nodes.begin();
	  ShallowArray<ZhengSORNExcUnit>::iterator end = _nodes.end();
	  for (; it != end; ++it) (*it).outputWeights(fsE2E, fsI2E);
	  fsE2E.close();
	  fsI2E.close();
	}
	++n;
	MPI::COMM_WORLD.Barrier();
      }
    }
  }
}

void ZhengSORNExcUnitCompCategory::inputWeightsShared(RNG& rng) 
{
  if (SHD.loadWeightsOn.size()>0) {
    if (SHD.loadWeightsOn[SHD.loadWeightsNext]==ITER) {
      if (SHD.loadWeightsOn.size()-1>SHD.loadWeightsNext) ++SHD.loadWeightsNext;
      int rank=getSimulation().getRank();
      int n=0; //node number
      // could get rid of the iteration because they can read all at the same time
	while (n<getSimulation().getNumProcesses()) {
	  if (n==rank) {
	    std::ifstream fsE2E, fsI2E, fsTEs;
	    std::ostringstream os;
	    os.str("");
	    // Import E2E weights
	    os<<"wee";
	    fsE2E.open(os.str().c_str(), std::ofstream::app|std::ofstream::out);
	    if (fsE2E.good()) {
	      //fsE2E.precision(8);
	      int row, col; float weight;
	      ShallowArray<ZhengSORNExcUnit>::iterator it;
	      while (!fsE2E.eof()) {
		fsE2E>>row>>col>>weight;
		ShallowArray<ZhengSORNExcUnit>::iterator end = _nodes.end();
		for (it = _nodes.begin(); it != end; ++it) {
		  if (it->getGlobalIndex()+1==row) {
		    it->inputWeights(fsE2E, col, weight);
		    break;
		  }
		}
	      }
	      fsE2E.close();
	    }
	    os.str(""); 
	    // Import I2E weights
	    os<<"wei";
	    fsI2E.open(os.str().c_str(), std::ofstream::app|std::ofstream::out);
	    if (fsI2E.good()) {
	      int row, col; float weight;
	      ShallowArray<ZhengSORNExcUnit>::iterator it2;
	      while (!fsI2E.eof()) {
		fsI2E>>row>>col>>weight;
		ShallowArray<ZhengSORNExcUnit>::iterator end2 = _nodes.end();
		for (it2 = _nodes.begin(); it2 != end2; ++it2) {
		  if (it2->getGlobalIndex()+1==row) {
		    it2->inputI2EWeights(fsI2E, col, weight);
		    break;
		  }
		}
	      }
	      fsI2E.close();
	    }
	  os.str("");
	  // Import thresholds
	  os<<"TEs";
	  fsTEs.open(os.str().c_str(), std::ofstream::app|std::ofstream::out);
	  if(fsTEs.good()){
	    int idx; float val;
	    ShallowArray<ZhengSORNExcUnit>::iterator it;
	    while (!fsTEs.eof()) {
		fsTEs>>idx>>val;
		ShallowArray<ZhengSORNExcUnit>::iterator end = _nodes.end();
		for (it = _nodes.begin(); it != end; ++it) {
		  if (it->getGlobalIndex()+1==idx) {
		    it->inputTE(val);
		    break;
		  }
		}
	    }
	    fsTEs.close();
	  }
	  os.str("");
      	}
	++n;
	MPI::COMM_WORLD.Barrier();
      }
    }
  }
}
