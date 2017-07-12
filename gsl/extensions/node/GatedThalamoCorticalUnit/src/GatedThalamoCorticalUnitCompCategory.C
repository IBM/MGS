#include "Lens.h"
#include "GatedThalamoCorticalUnitCompCategory.h"
#include "NDPairList.h"
#include "CG_GatedThalamoCorticalUnitCompCategory.h"
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


GatedThalamoCorticalUnitCompCategory::GatedThalamoCorticalUnitCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
   : CG_GatedThalamoCorticalUnitCompCategory(sim, modelName, ndpList)
{
}

void GatedThalamoCorticalUnitCompCategory::initializeShared(RNG& rng) 
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

void GatedThalamoCorticalUnitCompCategory::outputWeightsShared(RNG& rng) 
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
	  ShallowArray<GatedThalamoCorticalUnit>::iterator it = _nodes.begin();
	  ShallowArray<GatedThalamoCorticalUnit>::iterator end = _nodes.end();
	  for (; it != end; ++it) (*it).outputWeights(fsPH);
	  fsPH.close();
	}
	++n;
	MPI::COMM_WORLD.Barrier();
      }
    }
  }
}

void GatedThalamoCorticalUnitCompCategory::inputWeightsShared(RNG& rng) 
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
	  fsPH.open(os.str().c_str(), std::ofstream::app|std::ofstream::in); 
	  if (fsPH.good()) {
	    fsPH.precision(8);
	    int row, col;
	    ShallowArray<GatedThalamoCorticalUnit>::iterator it, end;
	    while (!fsPH.eof()) {
	      fsPH>>row>>col;
	      ShallowArray<GatedThalamoCorticalUnit>::iterator end = _nodes.end();
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
	MPI::COMM_WORLD.Barrier();
      }
    }
  }
}

void GatedThalamoCorticalUnitCompCategory::updateWhitMatrixShared(RNG& rng) 
{
#ifdef HAVE_ARMA  
  if (SHD.whitening && (ITER>1) && ((ITER % int(round(1.0/SHD.betaZ0))) == 0)) {
    ShallowArray<GatedThalamoCorticalUnit>::iterator it = _nodes.begin();
    ShallowArray<GatedThalamoCorticalUnit>::iterator end = _nodes.end();
    int totalNbrUnits = it->getGridLayerData()->getNbrUnits();
    std::string tempGridName = it->getGridLayerData()->getGridLayerDescriptor()->getGrid()->getName();
    //std::cout << "tempGridName: " << tempGridName << std::endl;
    for(; it!=end; it++){
      if(it->getGridLayerData()->getGridLayerDescriptor()->getGrid()->getName() != tempGridName){
        totalNbrUnits += it->getGridLayerData()->getNbrUnits();
        tempGridName = it->getGridLayerData()->getGridLayerDescriptor()->getGrid()->getName();
        //std::cout << "tempGridName: " << tempGridName << std::endl;
        }
    }
    //std::cout << "totalNbrUnits: " << totalNbrUnits << std::endl;
    
    int globalIdx_end = 0;
    int globalIdx_beg = 0;
    int localIdx = 0;
    // Area by area, retrieve covariance matrix and put in matrix structure (armadillo library) for inversion
    while(globalIdx_end<totalNbrUnits){
      globalIdx_beg = globalIdx_end;
      globalIdx_end += _nodes[localIdx].getGridLayerData()->getNbrUnits(); 
      
      // Write weights (gathered from nodes) [TODO: write/read in binary]
      std::ofstream ofs;
      std::ostringstream os;
      os<<"whit.bin";
      int rank=getSimulation().getRank();
      int n=0;
      while (n<getSimulation().getNumProcesses()) {
	if (n==rank) {
          if(n==0) ofs.open(os.str().c_str(), std::ofstream::trunc|std::ofstream::out); //|std::ofstream::binary);
          else ofs.open(os.str().c_str(), std::ofstream::app|std::ofstream::out); //|std::ofstream::binary);
          for (it=_nodes.begin(); it!=end; it++){
            if(it->getGlobalIndex() >= globalIdx_beg && it->getGlobalIndex() < globalIdx_end) {
	      it->getLateralCovInputs(ofs);
              localIdx++;
	    }
          }  
	  ofs.close(); 
   	}
        ++n;
	MPI::COMM_WORLD.Barrier(); 
      }
      // Read file, inverse sqrt of matrix, and set new weight back in node
      std::ifstream ifs("whit.bin", std::ifstream::in); // | std::ifstream::binary);
      int matSz = globalIdx_end - globalIdx_beg;
      mat q = mat(matSz, matSz);
      std::vector<double> vectorOfElem(matSz);
      for (int j=0; j<matSz; j++){
        for (int i=0; i<matSz; i++){
          ifs >> vectorOfElem[i];
        }
        q.row(j) = conv_to<rowvec>::from(vectorOfElem);
      }
      if (all(all(q))){
        cx_mat q_sqrt = sqrtmat(q); 
        cx_mat c = inv(q_sqrt);
        mat abs_c = abs(c);
        for (it=_nodes.begin(); it!=end; it++) {
          if(it->getGlobalIndex() >= globalIdx_beg && it->getGlobalIndex() < globalIdx_end) {
            std::vector<double> latWhitInputs = conv_to< std::vector<double> >::from(abs_c.row(it->getIndex()));
	    it->setLateralWhitInputs(&latWhitInputs);
          }
        }
      }
    }
  std::cout << "ITER:" << ITER << std::endl;
  }
#endif
}













