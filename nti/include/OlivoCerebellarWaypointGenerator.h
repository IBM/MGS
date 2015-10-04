// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// ================================================================

#ifndef OLIVOCEREBELLARWAYPOINTGENERATOR_H
#define OLIVOCEREBELLARWAYPOINTGENERATOR_H

#include "WaypointGenerator.h"
#include "VecPrim.h"
#include <string>
#include <algorithm>
#include "ShallowArray.h"

#define MIN_SQ_DIST_POINTS 100.0
#define MAX_MULTITHREADED_TRACTS 10
#define MAX_MULTITHREADED_NEURONS 250

class OlivoCerebellarWaypointGenerator : public WaypointGenerator
{
 private :

  typedef class point
  {
    public:
      point() {for (int i=0; i<3; ++i) coords[i]=0;}
      double coords[3];
  } point_t;

  typedef ShallowArray<point_t> tract_t;

  int _nrNeurons;
  ShallowArray<ShallowArray<tract_t, MAX_MULTITHREADED_TRACTS, 4 >, MAX_MULTITHREADED_NEURONS, 4 > _allNeuronTracts;
  ShallowArray<ShallowArray<int, MAX_MULTITHREADED_TRACTS, 4 >, MAX_MULTITHREADED_NEURONS, 4 > _pointIndexPerNeuronTract;
  ShallowArray<int> _maxTractPointsPerNeuron;
  ShallowArray<int> _nextPerNeuron;
  
 public :
  OlivoCerebellarWaypointGenerator() : _nrNeurons(0)
  {
  }
    
  void readWaypoints(char** fileNames, int nrNeurons) 
  {
    _nrNeurons=nrNeurons;
    _allNeuronTracts.increaseSizeTo(_nrNeurons);
    _pointIndexPerNeuronTract.increaseSizeTo(_nrNeurons);
    _maxTractPointsPerNeuron.increaseSizeTo(_nrNeurons);
    _nextPerNeuron.increaseSizeTo(_nrNeurons);
    for (int i=0; i<_nrNeurons; ++i) _nextPerNeuron[i]=0;

    std::string nextLine;
	
    for (int nid=0; nid<nrNeurons; ++nid){
      ShallowArray<tract_t, MAX_MULTITHREADED_TRACTS, 4>& tractsVec=_allNeuronTracts[nid];
		
      std::string fileName(fileNames[nid]);
      fileName = fileName.substr(0,fileName.length()-3);	
      fileName.append("cft");
      std::ifstream tractFile (fileName.c_str());
     
      point_t nextPoint, lastPoint;
      int nr_tracts=0;
      double running_dist=0;
      bool tractTerminated=true;
      if (tractFile.is_open()) {
	while ( tractFile.good() && getline(tractFile, nextLine) ) {
	  if (nextLine!="" && nextLine[0]!='#') {
	    if (nextLine.find('.')==std::string::npos) { // an integer providing the number of points
	      if (!tractTerminated) tractsVec[nr_tracts-1].push_back(lastPoint);
	      ++nr_tracts;
	      tract_t nextTract;
	      tractsVec.push_back(nextTract);
	      running_dist=0;
	    }
	    else {
	      tract_t& thisTract=tractsVec[nr_tracts-1];
	      std::istringstream iss(nextLine);
	      int nr_tract_pts=thisTract.size();
	      // Add point to a new tract, else add it to the last tract but not too close to the last point. 
	      for (int i=0; i<3; ++i) iss>>nextPoint.coords[i];
	      if (nr_tract_pts==0) {
		thisTract.push_back(nextPoint);
		tractTerminated=true;
	      }
	      else if (running_dist>MIN_SQ_DIST_POINTS) {
		thisTract.push_back(nextPoint);
		tractTerminated=true;
		running_dist=0;
	      }
	      else {
		running_dist+=SqDist(lastPoint.coords, nextPoint.coords);
		tractTerminated=false;
	      }
	      lastPoint=nextPoint;
	    }
	  }
	}
      }
      if (nr_tracts>0) {
	if (!tractTerminated) tractsVec[nr_tracts-1].push_back(lastPoint);
	tractFile.close();

	// Update the maxSize of the longest tract
	_maxTractPointsPerNeuron[nid] = tractsVec[0].size();
	for (int i=1; i<tractsVec.size(); ++i) {
	  if (tractsVec[i].size() > _maxTractPointsPerNeuron[nid])
	    _maxTractPointsPerNeuron[nid] = tractsVec[i].size();			
	}
	// Start paths at different point indices to reflect branching patter
	_pointIndexPerNeuronTract[nid].increaseSizeTo(nr_tracts);
	_pointIndexPerNeuronTract[nid][0]=0;
	for (int i=1; i<nr_tracts; ++i) {
	  _pointIndexPerNeuronTract[nid][i]=_pointIndexPerNeuronTract[nid][i-1]+1+int( (i==1 ? 0.25 : 0.02)*(double(tractsVec[i].size() ) ) );
	  if (_pointIndexPerNeuronTract[nid][i]>tractsVec[i].size()-1) {
	    point_t tmp=tractsVec[i][tractsVec[i].size()-1];
	    tractsVec[i].increaseSizeTo(_pointIndexPerNeuronTract[nid][i]+1);
	    tractsVec[i][_pointIndexPerNeuronTract[nid][i]]=tmp;
	  }	  
	}
      }
    }
  }
  
  
  void next(ShallowArray<ShallowArray<double> >& waypointCoords, int nid)
  {
    ShallowArray<tract_t, MAX_MULTITHREADED_TRACTS, 4>& tractsVec = _allNeuronTracts[nid];
    int ntracts=tractsVec.size();

    waypointCoords.clear();
    waypointCoords.increaseSizeTo(ntracts);

    if (_nextPerNeuron[nid]<_maxTractPointsPerNeuron[nid]) {
      for (int tractIndex=0; tractIndex<ntracts; ++tractIndex) {
	int& nextPointIndex=_pointIndexPerNeuronTract[nid][tractIndex];
	if (_nextPerNeuron[nid]>=nextPointIndex && nextPointIndex<tractsVec[tractIndex].size()){	  
	  double* cds=tractsVec[tractIndex][nextPointIndex].coords;
	  for (int i=0; i<3; ++i) waypointCoords[tractIndex].push_back(cds[i]);
	  /*
	  int rank;
	  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	  printf("rank %d, nid %d, tractIndex %d, pointIndex %d, %f, %f, %f\n", rank, nid, tractIndex, nextPointIndex, cds[0], cds[1], cds[2]); 
	  */
	  ++nextPointIndex;
	}
      }
    }
    ++_nextPerNeuron[nid];
  }

  void reset() 
  {
    for (int i=0; i<_nrNeurons; ++i) {
      _nextPerNeuron[i]=0;
      for (int j=0; j<_pointIndexPerNeuronTract[i].size(); ++j) {
	_pointIndexPerNeuronTract[i][j]=0;
      }
    }
  }
  
  ~OlivoCerebellarWaypointGenerator() 
  {
  }
  
};
#endif
