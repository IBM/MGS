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
#include "ZhengSORNInhUnitCompCategory.h"
#include "NDPairList.h"
#include "CG_ZhengSORNInhUnitCompCategory.h"
#include <fstream>
#include <sstream>
#include <iostream>

#define SHD getSharedMembers()
#define ITER getSimulation().getIteration()

ZhengSORNInhUnitCompCategory::ZhengSORNInhUnitCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
   : CG_ZhengSORNInhUnitCompCategory(sim, modelName, ndpList)
{
}


void ZhengSORNInhUnitCompCategory::initializeShared(RNG& rng) 
{
  int n=SHD.collectWeightsOn.size();
  if (n>0) {
    for (int i=0; i<n; ++i) {
      std::ofstream fs; std::ostringstream os;
      os.str("");
      os<<SHD.outDirectory<<"E2I_"<<SHD.outputWeightsFileName<<"_"<<SHD.collectWeightsOn[i]<<".dat";
      fs.open(os.str().c_str(), std::ofstream::trunc|std::ofstream::out);
      fs.close();
    }
  }
}

void ZhengSORNInhUnitCompCategory::outputWeightsShared(RNG& rng) 
{
  if (SHD.collectWeightsOn.size()>0) {
    if (SHD.collectWeightsOn[SHD.collectWeightsNext]==ITER) {
      if (SHD.collectWeightsOn.size()-1>SHD.collectWeightsNext) ++SHD.collectWeightsNext;
      int rank=getSimulation().getRank();
      int n=0;
      while (n<getSimulation().getNumProcesses()) {
	if (n==rank) {
	  std::ofstream fsE2I; 
	  std::ostringstream os;
	  os.str("");
          os<<SHD.outDirectory<<"E2I_"<<SHD.outputWeightsFileName<<"_"<<ITER<<".dat";
	  fsE2I.open(os.str().c_str(), std::ofstream::app|std::ofstream::out);
	  ShallowArray<ZhengSORNInhUnit>::iterator it = _nodes.begin();
	  ShallowArray<ZhengSORNInhUnit>::iterator end = _nodes.end();
	  for (; it != end; ++it) (*it).outputWeights(fsE2I);
	  fsE2I.close();
	}
	++n;
	MPI_Barrier(MPI_COMM_WORLD);
      }
    }
  }
}

void ZhengSORNInhUnitCompCategory::inputWeightsShared(RNG& rng) 
{
  if (SHD.loadWeightsOn.size()>0) {
    if (SHD.loadWeightsOn[SHD.loadWeightsNext]==ITER) {
      if (SHD.loadWeightsOn.size()-1>SHD.loadWeightsNext) ++SHD.loadWeightsNext;
      int rank=getSimulation().getRank();
      int n=0; //node number
      // could get rid of the iteration because they can read all at the same time
      while (n<getSimulation().getNumProcesses()) {
	if (n==rank) {
	  std::ifstream fsE2I, fsTIs, fsYs;
	  std::ostringstream os;
	  os.str("");
	  os<<SHD.inDirectory<<SHD.inFiles[0]; //"wie";
	  fsE2I.open(os.str().c_str(), std::ofstream::app|std::ofstream::out);
	  if (fsE2I.good()) {
	    int row, col; float weight;
	    ShallowArray<ZhengSORNInhUnit>::iterator it;//, end;
	    while (!fsE2I.eof()) {
	      fsE2I>>row>>col>>weight;
	      ShallowArray<ZhengSORNInhUnit>::iterator end = _nodes.end();
	      for (it = _nodes.begin(); it != end; ++it) {
		if (it->getGlobalIndex()+1==row) {
		  it->inputWeights(fsE2I, col, weight);
		  break;
		}
	      }
	    }
	    fsE2I.close();
	  }
	  os.str("");
	  
          // Import thresholds
	  if (SHD.inFiles.size()>1) {
	    os<<SHD.inDirectory<<SHD.inFiles[1]; //"TIs";
	    fsTIs.open(os.str().c_str(), std::ofstream::app|std::ofstream::out);
	    if(fsTIs.good()){
	      int idx; float val;
	      ShallowArray<ZhengSORNInhUnit>::iterator it;
	      while (!fsTIs.eof()) {
		  fsTIs>>idx>>val;
		  ShallowArray<ZhengSORNInhUnit>::iterator end = _nodes.end();
		  for (it = _nodes.begin(); it != end; ++it) {
		    if (it->getGlobalIndex()+1==idx) {
		      it->inputTI(val);
		      break;
		    }
		  }
	        }
	      fsTIs.close();
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


