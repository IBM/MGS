// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Mgs.h"
#include "NazeSORNExcUnitCompCategory.h"
#include "NDPairList.h"
#include "CG_NazeSORNExcUnitCompCategory.h"
#include <fstream>
#include <sstream>
#include <iostream>

#define SHD getSharedMembers()
#define ITER getSimulation().getIteration()

NazeSORNExcUnitCompCategory::NazeSORNExcUnitCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
   : CG_NazeSORNExcUnitCompCategory(sim, modelName, ndpList)
{
}

void NazeSORNExcUnitCompCategory::initializeShared(RNG& rng) 
{
  SHD.eta_iSTDP = (1.0+1.0/SHD.mu_iSTDP);
  SHD.eta_iLTP = SHD.eta_inhib * (1.0-SHD.eta_iSTDP);
  SHD.sigma_HIP = SHD.mu_HIP*SHD.ratio_HIP;
  SHD.sigma_IP = SHD.mu_IP*SHD.ratio_IP;
  SHD.eta_STDP = 4*SHD.eta_inhib;
  //SHD.sigma_delay = SHD.mu_delay*SHD.ratio_delay;
  int n=SHD.collectWeightsOn.size();
  if (n>0) {
    for (int i=0; i<n; ++i) {
      std::ofstream fsE2E, fsI2E; std::ostringstream os;
      os.str("");
      os<<SHD.outDirectory<<"E2E_"<<SHD.outputWeightsFileName<<"_"<<SHD.collectWeightsOn[i]<<".dat";
      fsE2E.open(os.str().c_str(), std::ofstream::trunc|std::ofstream::out);
      fsE2E.close();
      os.str(""); 
      os<<SHD.outDirectory<<"I2E_"<<SHD.outputWeightsFileName<<"_"<<SHD.collectWeightsOn[i]<<".dat";
      fsI2E.open(os.str().c_str(), std::ofstream::trunc|std::ofstream::out);
      fsI2E.close();
    }
  }
}

void NazeSORNExcUnitCompCategory::saveInitParams(RNG& rng)
{
  std::ofstream fs_etaIP, fs_HIP;
  std::ostringstream os;
  os.str("");
  os << SHD.outDirectory << "etaIPs.txt";
  fs_etaIP.open(os.str().c_str(), std::ofstream::trunc|std::ofstream::out);
  os.str("");
  os << SHD.outDirectory << "HIPs.txt";
  fs_HIP.open(os.str().c_str(), std::ofstream::trunc|std::ofstream::out);

  ShallowArray<NazeSORNExcUnit>::iterator it = _nodes.begin();
  ShallowArray<NazeSORNExcUnit>::iterator end = _nodes.end();
  for(; it!=end; ++it) {
    (*it).getInitParams(fs_etaIP, fs_HIP);
  }
  fs_etaIP << std::endl;
  fs_HIP << std::endl; 
  fs_etaIP.close();
  fs_HIP.close();
  
}

void NazeSORNExcUnitCompCategory::outputWeightsShared(RNG& rng) 
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
	  os<<SHD.outDirectory<<"E2E_"<<SHD.outputWeightsFileName<<"_"<<ITER<<".dat";
	  fsE2E.open(os.str().c_str(), std::ofstream::app|std::ofstream::out);
	  os.str("");
	  os<<SHD.outDirectory<<"I2E_"<<SHD.outputWeightsFileName<<"_"<<ITER<<".dat";
	  fsI2E.open(os.str().c_str(), std::ofstream::app|std::ofstream::out);
	  
          ShallowArray<NazeSORNExcUnit>::iterator it = _nodes.begin();
	  ShallowArray<NazeSORNExcUnit>::iterator end = _nodes.end();
	  for (; it != end; ++it) (*it).outputWeights(fsE2E, fsI2E);
	  fsE2E.close();
	  fsI2E.close();
	}
	++n;
	MPI_Barrier(MPI_COMM_WORLD);
      }
    }
  }
}

void NazeSORNExcUnitCompCategory::inputWeightsShared(RNG& rng) 
{
  if (SHD.loadWeightsOn.size()>0) {
    if (SHD.loadWeightsOn[SHD.loadWeightsNext]==ITER) {
      std::cout << "Importing SORN Weights at timestep " << ITER << std::endl;
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
	    os<<SHD.inDirectory<<SHD.inFiles[0]; //"wee"
	    fsE2E.open(os.str().c_str(), std::ofstream::app|std::ofstream::out);
	    if (fsE2E.good()) {
	      //fsE2E.precision(8);
	      int row, col; float weight;
	      ShallowArray<NazeSORNExcUnit>::iterator it;
	      while (!fsE2E.eof()) {
		fsE2E>>row>>col>>weight;
		ShallowArray<NazeSORNExcUnit>::iterator end = _nodes.end();
		for (it = _nodes.begin(); it != end; ++it) {
		  if (it->getGlobalIndex()+1==row) {
		    it->inputWeights(col, weight);
		    break;
		  }
		}
	      }
	      fsE2E.close();
	    }
	    os.str(""); 
	    // Import I2E weights
	    os<<SHD.inDirectory<<SHD.inFiles[1]; //"wei"
	    fsI2E.open(os.str().c_str(), std::ofstream::app|std::ofstream::out);
	    if (fsI2E.good()) {
	      int row, col; float weight;
	      ShallowArray<NazeSORNExcUnit>::iterator it2;
	      while (!fsI2E.eof()) {
		fsI2E>>row>>col>>weight;
		ShallowArray<NazeSORNExcUnit>::iterator end2 = _nodes.end();
		for (it2 = _nodes.begin(); it2 != end2; ++it2) {
		  if (it2->getGlobalIndex()+1==row) {
		    it2->inputI2EWeights(col, weight);
		    break;
		  }
		}
	      }
	      fsI2E.close();
	    }
	  os.str("");
	  
          if (SHD.inFiles.size()>2) {
	    // Import thresholds
	    os<<SHD.inDirectory<<SHD.inFiles[2]; //"TEs"
	    fsTEs.open(os.str().c_str(), std::ofstream::app|std::ofstream::out);
	    if(fsTEs.good()){
	      int idx; float val;
	      ShallowArray<NazeSORNExcUnit>::iterator it;
	      while (!fsTEs.eof()) {
		  fsTEs>>idx>>val;
		  ShallowArray<NazeSORNExcUnit>::iterator end = _nodes.end();
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
      	}
	++n;
	MPI_Barrier(MPI_COMM_WORLD);
      }
    }
  }
}

void NazeSORNExcUnitCompCategory::outputDelaysShared(RNG& rng) 
{
  if (SHD.collectDelaysOn.size()>0) {
    if (SHD.collectDelaysOn[SHD.collectDelaysNext]==ITER) {
      if (SHD.collectDelaysOn.size()-1>SHD.collectDelaysNext) ++SHD.collectDelaysNext;
      int rank=getSimulation().getRank();
      int n=0;
      while (n<getSimulation().getNumProcesses()) {
	if (n==rank) {
	  std::ofstream fsE2Ed; 
	  std::ostringstream os;
	  os<<SHD.outDirectory<<"E2E_"<<SHD.outputDelaysFileName<<"_"<<ITER<<".dat";
	  fsE2Ed.open(os.str().c_str(), std::ofstream::app|std::ofstream::out);

          ShallowArray<NazeSORNExcUnit>::iterator it = _nodes.begin();
	  ShallowArray<NazeSORNExcUnit>::iterator end = _nodes.end();
	  for (; it != end; ++it) (*it).outputDelays(fsE2Ed);
	  fsE2Ed.close();
	}
	++n;
	MPI_Barrier(MPI_COMM_WORLD);
      }
    }
  }
}
