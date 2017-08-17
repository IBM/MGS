// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifdef HAVE_MPI
#include <mpi.h>
#include "BGCartesianPartitioner.h"
#include <fstream>
#include <iostream>
#include <cassert>
#include <math.h>
#include <list>
#include "Granule.h"
#ifdef USING_BLUEGENEL
#include <rts.h>
#else
#ifdef USING_BLUEGENEP
#include <spi/kernel_interface.h>
#include <common/bgp_personality.h>
#include <common/bgp_personality_inlines.h>
#else
#include "VolumeDivider.h"
#endif
#endif
#include "VectorOstream.h"

#define DIM0 4
#define DIM1 4
#define DIM2 4
//#define INSTRUMENT

/* BGCartesianPartitioner makes uses of coordinates assigned to each granule to describe the granules location in the
   dataspace from which it was mapped. It assumes these coordinates have been assigned such that the granule coordinate
   system is rotated such that the first coordinate is from the shortest dimension, and the last is from the longest */

BGCartesianPartitioner::BGCartesianPartitioner(const std::string& fileName, bool outputGraph, Simulation* sim)
  : Partitioner(), _fileName(fileName), _outputGraph(outputGraph), _sim(sim), _nprocs(0)
{
  // Set up mesh
#ifdef USING_BLUEGENEL
  BGLPersonality personality;
  rts_get_personality(&personality, sizeof(personality));
  _mesh[0]=personality.xSize;
  _mesh[1]=personality.ySize;
  _mesh[2]=personality.zSize;
#else
#if USING_BLUEGENEP
  _BGP_Personality_t personality;
  Kernel_GetPersonality(&personality, sizeof(personality));
  int node_config = personality.Kernel_Config.ProcessConfig;
  if (node_config == _BGP_PERS_PROCESSCONFIG_SMP) printf("SMP mode\n\n");
  else if (node_config == _BGP_PERS_PROCESSCONFIG_VNM) printf("Virtual-node mode\n\n");
  else if (node_config == _BGP_PERS_PROCESSCONFIG_2x2) printf("Dual mode\n\n");
  else printf ("Unknown mode\n\n");
  _mesh[0]=personality.Network_Config.Xnodes;
  _mesh[1]=personality.Network_Config.Ynodes;
  _mesh[2]=personality.Network_Config.Znodes;  
#else
  _mesh[0]=DIM0;
  _mesh[1]=DIM1;
  _mesh[2]=DIM2;
#endif
#endif

  // Find mesh ordering
  while (_meshOrder.size()<DIM) {
    int min=INT_MAX;
    int minIdx=-1;
    for (int i = 0; i < DIM; ++i) {
      if (_mesh[i]<min) {
	bool replaceMin=true;
	for (int j = 0; j < _meshOrder.size(); ++j) {
	  if (i==_meshOrder[j]) {
	    replaceMin=false;
	    break;
	  }
	}
	if (replaceMin) {
	  min=_mesh[i];
	  minIdx=i;
	}
      }
    }
    _meshOrder.push_back(minIdx);
  }

  // Set up coordinate to ranks mapping
  MPI_Comm cart_comm;
  _nprocs = 1;
  for (int i=0; i<DIM; ++i) _nprocs*=_mesh[i];
  int periods[DIM] = {0,0,0};
  int reorder = 0;
}

void BGCartesianPartitioner::partition(std::vector<Granule*>& graph, 
				      unsigned numberOfPartitions)
{
  assert (numberOfPartitions==_nprocs);

  int*** ranks=new int**[_mesh[0]];
  for (int i=0; i<_mesh[0]; ++i) {
    ranks[i]=new int*[_mesh[1]];
    for (int j=0; j<_mesh[1]; ++j) {
      ranks[i][j]=new int[_mesh[2]];
    }
  }
 int rank=0, coords[DIM];
  for (int i=0; i<_mesh[0]; ++i) {
    for (int j=0; j<_mesh[1]; ++j) {
      for (int k=0; k<_mesh[2]; ++k) {
	MPI_Cart_coords(MPI_COMM_WORLD, rank, DIM, coords);
#ifdef INSTRUMENT
	std::cerr<<"rank "<<rank<<", "<<coords[0]<<", "<<coords[1]<<", "<<coords[2]<<std::endl;
#endif
	ranks[coords[0]][coords[1]][coords[2]]=rank;
	++rank;
      }
    }
  }

  // Set up output file
  std::vector<Granule*>::iterator it, end = graph.end(); 
  std::ofstream* file=0;
  if (_outputGraph) file=new std::ofstream(_fileName.c_str());


  // Set up mesh histograms
  int** meshHist = new int*[DIM];
  for (int i=0; i<DIM; ++i) {
    int sz=_mesh[i];
    meshHist[i]=new int[sz];
    for (int j=0; j<sz; ++j) {
      meshHist[i][j]=0;
    }
  }

  // initialize slice map (and enumerate mesh dimensions)
  std::map<double, int> meshCoordinates[DIM];
  for (it = graph.begin(); it != end; ++it) {
    Granule* g = (*it);
    std::vector<double>& cds=g->getModifiableGranuleCoordinates();    
    for (int i=0; i<DIM; ++i) {
      int dim=_meshOrder[i];
      meshCoordinates[dim][cds[i]]=0;
      meshHist[dim][int(floor(double(_mesh[dim])*cds[i]))]++;
    }
  }


  // Ensure BGCartesianPartitioner assumptions satisfied (_nprocs>=ngranules)
  int checkSize=1;
  for (int i=0; i<DIM; ++i) {
    checkSize*=meshCoordinates[i].size();
  }
  assert (checkSize>=_nprocs);

#ifdef INSTRUMENT
  for (int i=0; i<DIM; ++i) {
    for (int j=0; j<_mesh[i]; ++j) {
      std::cerr<<"meshHist : "<<i<<", "<<j<<" : "<<meshHist[i][j]<<std::endl;
    }
  }
#endif

  // determine how many granules to deal out to each dimension 
  std::vector<int> meshOrig[DIM];  
  std::vector<int> meshDeal[DIM];
  for (int i=1; i<DIM; ++i) {
    int count=1, idx=0, orig=0;
   while (++idx<_mesh[i]) {
      while (idx<_mesh[i] && meshHist[i][idx]==0) {
	++count;
	++idx;
      }
      if (count>1) {
	meshOrig[i].push_back(orig);
	meshDeal[i].push_back(count);
	count=1;
      }
      orig=idx;
    }
    if (count>1) {
      meshOrig[i].push_back(orig);
      meshDeal[i].push_back(count);
    }
  }

#ifdef INSTRUMENT
  for (int i=1; i<DIM; ++i) {
    for (int j=0; j<meshDeal[i].size(); ++j)
      std::cerr<<"meshOrig["<<i<<"]["<<j<<"] : "<<meshOrig[i][j]<<std::endl;
    for (int j=0; j<meshDeal[i].size(); ++j)
      std::cerr<<"meshDeal["<<i<<"]["<<j<<"] : "<<meshDeal[i][j]<<std::endl;
  }
#endif

  // deal out granules in the slice dimension to the slice map
  std::map<double, int>::iterator mapIter;
  std::vector<double> cds[DIM];
  for (int i=0; i<DIM; ++i) {
    for (mapIter=meshCoordinates[i].begin(); mapIter!=meshCoordinates[i].end(); ++mapIter) { 
      cds[i].push_back(mapIter->first);
#ifdef INSTRUMENT
      std::cerr<<"cds["<<i<<"]["<<cds[i].size()-1<<"] : "<<cds[i][cds[i].size()-1]<<std::endl;
#endif
    }
  }

  std::map<int, std::map<double, int> > sliceMap[DIM]; 
  for (int i=1; i<DIM; ++i) {
    for (int j=0; j<meshOrig[i].size(); ++j) {
      sliceMap[i][meshOrig[i][j]]=meshCoordinates[i];
      for (int k=0; k<cds[0].size(); ++k) {
	if (i==1 || meshOrig[1].size()==0) 
	  sliceMap[i][meshOrig[i][j]][cds[0][k]]=k%meshDeal[i][j];
	else if (i==2) {
	  if (k==0) assert(cds[0].size()/meshDeal[1][j]/meshDeal[2][j]>=_mesh[2]); // must be so to populate checkerboard fully
	  sliceMap[i][meshOrig[i][j]][cds[0][k]]=k/meshDeal[i][j];
	}
#ifdef INSTRUMENT
	if (i==1 || meshOrig[1].size()==0)
	  std::cerr<<"sliceMap["<<i<<"]["<<meshOrig[i][j]<<"]["<<cds[0][k]<<"]="<<k%meshDeal[i][j]<<std::endl;
	else if (i==2)
	  std::cerr<<"sliceMap["<<i<<"]["<<meshOrig[i][j]<<"]["<<cds[0][k]<<"]="<<k/meshDeal[i][j]<<std::endl;
#endif
      }
    }
  }

#ifdef INSTRUMENT
  for (int i=0; i<DIM; ++i) {
    for (int j=0; j<_mesh[i]; ++j) {
      meshHist[i][j]=0;
    }
  }
#endif   
  
  // determine rank coordinate
  int rankCds[DIM];
  int count=0;
  std::map<int, std::map<double, int> >::iterator mapIter2;
  for (it = graph.begin(); it != end; ++it, ++count) {
    Granule* g = (*it);
    std::vector<double>& gcds=g->getModifiableGranuleCoordinates();
    for (int i=0; i<DIM; ++i) {
      int dim=_meshOrder[i];
      rankCds[dim]=int(floor(double(_mesh[dim])*gcds[i]));
      if (dim!=0) {
	mapIter2=sliceMap[dim].find(rankCds[dim]);
	if (mapIter2!=sliceMap[dim].end()) {
	  mapIter=mapIter2->second.find(gcds[0]);
	  if (mapIter!=mapIter2->second.end()) {
	    rankCds[dim]+=mapIter->second;
	  }
	}
      }
#ifdef INSTRUMENT
      meshHist[dim][rankCds[dim]]++;
#endif
    }
    
    // set rank
    g->setPartitionId( ranks[rankCds[0]][rankCds[1]][rankCds[2]] );
    if (_outputGraph) {
      (*file) << *(*it);
      (*file) << "\n";
    }
  }

#ifdef INSTRUMENT
  for (int i=0; i<DIM; ++i) {
    for (int j=0; j<_mesh[i]; ++j) {
      std::cerr<<"meshHist modified : "<<i<<", "<<j<<" : "<<meshHist[i][j]<<std::endl;
    }
  }
#endif

  if (_outputGraph) file->close();
  delete file;
  for (int i=0; i<_mesh[0]; ++i) {
    for (int j=0; j<_mesh[1]; ++j) {
      delete [] ranks[i][j];
    }
    delete [] ranks[i];
  }
  delete [] ranks;

  for (int i=0; i<DIM; ++i) {
    delete [] meshHist[i];
  }
  delete [] meshHist;
}
#endif
