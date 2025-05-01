// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Mgs.h"
#include "LinskerInfomaxUnitCompCategory.h"
#include "NDPairList.h"
#include "CG_LinskerInfomaxUnitCompCategory.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#ifdef HAVE_ARMA
#include <armadillo>
using namespace arma;
#endif

#define SHD getSharedMembers()
#define ITER getSimulation().getIteration()

LinskerInfomaxUnitCompCategory::LinskerInfomaxUnitCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
   : CG_LinskerInfomaxUnitCompCategory(sim, modelName, ndpList)
{
}

void LinskerInfomaxUnitCompCategory::initializeShared(RNG& rng) 
{
  int n=SHD.collectWeightsOn.size();
  if (n>0) {
    std::ofstream fs; std::ostringstream os;
    for (int i=0; i<n; ++i) {
      os.str("");
      os<<"IMAX_TH_"<<SHD.weightsFileName<<"_"<<SHD.collectWeightsOn[i]<<".dat";
      fs.open(os.str().c_str(), std::ofstream::trunc|std::ofstream::out);
      fs.close();

      os.str("");
      os<<"IMAX_LN_"<<SHD.weightsFileName<<"_"<<SHD.collectWeightsOn[i]<<".dat";
      fs.open(os.str().c_str(), std::ofstream::trunc|std::ofstream::out);
      fs.close();
    }
  }
}

void LinskerInfomaxUnitCompCategory::outputWeightsShared(RNG& rng) 
{
  if (SHD.collectWeightsOn.size()>0) {
    if (SHD.collectWeightsOn[SHD.collectWeightsNext]==ITER) {
      if (SHD.collectWeightsOn.size()-1>SHD.collectWeightsNext) ++SHD.collectWeightsNext;
      int rank=getSimulation().getRank();
      int n=0;
      while (n<getSimulation().getNumProcesses()) {
	if (n==rank) {
	  std::ofstream fsTH, fsLN; 
	  std::ostringstream os;
	  os<<"IMAX_TH_"<<SHD.weightsFileName<<"_"<<ITER<<".dat";
	  fsTH.open(os.str().c_str(), std::ofstream::app|std::ofstream::out);
	  os.str("");
	  os<<"IMAX_LN_"<<SHD.weightsFileName<<"_"<<ITER<<".dat";
	  fsLN.open(os.str().c_str(), std::ofstream::app|std::ofstream::out);
	  ShallowArray<LinskerInfomaxUnit>::iterator it = _nodes.begin();
	  ShallowArray<LinskerInfomaxUnit>::iterator end = _nodes.end();
	  for (; it != end; ++it) (*it).outputWeights(fsTH, fsLN);
	  fsTH.close();
	  fsLN.close();
	}
	++n;
	MPI_Barrier(MPI_COMM_WORLD);
      }
    }
  }
}

void LinskerInfomaxUnitCompCategory::invertQmatrixShared(RNG& rng) {
#ifdef HAVE_ARMA
  if (SHD.inversion_method && (ITER % SHD.period)==0){
    ShallowArray<LinskerInfomaxUnit>::iterator it = _nodes.begin();
    ShallowArray<LinskerInfomaxUnit>::iterator end = _nodes.end();
    int totalNbrUnits = it->getGridLayerData()->getNbrUnits();
    std::string tempGridName = it->getGridLayerData()->getGridLayerDescriptor()->getGrid()->getName();
    for(; it!=end; it++){
      if(it->getGridLayerData()->getGridLayerDescriptor()->getGrid()->getName() != tempGridName){
        totalNbrUnits += it->getGridLayerData()->getNbrUnits();
        tempGridName = it->getGridLayerData()->getGridLayerDescriptor()->getGrid()->getName();
        //std::cout << "tempGridName: " << tempGridName << std::endl;
        }
    }
  
    int globalIdx_end = 0;
    int globalIdx_beg = 0;
    int localIdx = 0;
    while(globalIdx_end<totalNbrUnits){
      globalIdx_beg = globalIdx_end;
      globalIdx_end += _nodes[localIdx].getGridLayerData()->getNbrUnits();;
      
      int np=getSimulation().getNumProcesses();
      if (np>1) {
	std::ofstream ofsW;
	std::ostringstream osW;
	osW<<"LinskerW.bin";
	
	int rank=getSimulation().getRank();
	int n=0;
	while (n<getSimulation().getNumProcesses()) {
	  if (n==rank) {
	    if(n==0) {
	      ofsW.open(osW.str().c_str(), std::ofstream::trunc|std::ofstream::out); //|std::ofstream::binary);
	    } else {
	      ofsW.open(osW.str().c_str(), std::ofstream::app|std::ofstream::out); //|std::ofstream::binary);
	    }
	    for (it=_nodes.begin(); it!=end; it++){
	      if(it->getGlobalIndex() >= globalIdx_beg && it->getGlobalIndex() < globalIdx_end) {
		it->getInputWeights(ofsW);
		localIdx++;
		//std::cout << n << "_" << it->getGlobalIndex() << " wrote to file" << std::endl;
	      }
	    }  
	    ofsW.close(); 
	  }
	  ++n;
	  MPI_Barrier(MPI_COMM_WORLD);
	}
	
	// populate matrix row by row by accessing node objects
	std::ifstream ifsW("LinskerW.bin", std::ifstream::in); // | std::ifstream::binary);
	int matSz = globalIdx_end - globalIdx_beg;
	mat W = mat(matSz, matSz);
	std::vector<double> vectorOfW(matSz);
	for (int j=0; j<matSz; j++){
	  for (int i=0; i<matSz; i++){
	    ifsW >> vectorOfW[i];
	  }
	  W.row(j) = conv_to<rowvec>::from(vectorOfW);
	}
	
	// invert matrix (pseudo-inverse)
	mat Wt = W.t();
	W = pinv(Wt);
	// save replace new weigths
	for (it=_nodes.begin(); it!=end; it++) {
	  if(it->getGlobalIndex() >= globalIdx_beg && it->getGlobalIndex() < globalIdx_end) {
	    std::vector<double> Wrow = conv_to< std::vector<double> >::from(W.row(it->getIndex()));
	    it->setInputWeights(&Wrow);
	    //std::cout << n << "_" << it->getGlobalIndex() << " weights updated" << std::endl;
	  }
	}
      } else {
        // on one process, no need to read/write on disk
        int matSz = globalIdx_end - globalIdx_beg;
	mat W = mat(matSz, matSz);
        std::vector<double> W_j(matSz);
        int idx=0;
        for (it=_nodes.begin(); it!=end; it++){
	  if(it->getGlobalIndex() >= globalIdx_beg && it->getGlobalIndex() < globalIdx_end) {
	    it->getInputWeights(&W_j);
            W.row(idx) =  conv_to<rowvec>::from(W_j);
	    idx++;
	    localIdx++;
	  }
	} 
      	// invert matrix (pseudo-inverse)
	if (any(any(W))){
	  mat Wt = W.t();
	  W = pinv(Wt);
	}
	// save replace new weigths
	for (it=_nodes.begin(); it!=end; it++) {
	  if(it->getGlobalIndex() >= globalIdx_beg && it->getGlobalIndex() < globalIdx_end) {
	    std::vector<double> Wrow = conv_to< std::vector<double> >::from(W.row(it->getIndex()));
	    it->setInputWeights(&Wrow);
	  }
        }
        if ((ITER % 10000)==0) {
          std::cout << "ITER = " << ITER << "W inverted" << std::endl;
        }
      }
    }
  }
#endif
}
