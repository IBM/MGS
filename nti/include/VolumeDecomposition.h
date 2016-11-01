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

#ifndef VOLUMEDECOMPOSITION_H
#define VOLUMEDECOMPOSITION_H

#include "Decomposition.h"
#include "Tissue.h"
#include "Sphere.h"
#include "RunTimeTopology.h"
#include <mpi.h>

#include "ShallowArray.h"
#include <vector>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

class Tissue;
class Sphere;
class Communicator;
class TouchSpace;

class VolumeDecomposition : public Decomposition
{
 public:
  VolumeDecomposition(int rank, FILE* inputFile, const int numVolumes, 
		Tissue* tissue, int X, int Y, int Z);
  VolumeDecomposition(VolumeDecomposition&);
  virtual ~VolumeDecomposition();
  Decomposition* duplicate();

  void getRanks(Sphere* sphere, double* coords2, double deltaRadius, ShallowArray<int, MAXRETURNRANKS, 100>& ranks);
  void addRanks(Sphere* sphere, double* coords2, double deltaRadius, ShallowArray<int, MAXRETURNRANKS, 100>& ranks);
  bool mapsToRank(double* coords,double radius, int rank);
  bool mapsToRank(Sphere* sphere, double* coords2, double deltaRadius, int rank);

  void prepareToDecompose();
  void decompose();
  void getRanks(Sphere* sphere, double deltaRadius, ShallowArray<int, MAXRETURNRANKS, 100>& ranks);
  int getRank(Sphere& sphere);
  int getRank(double* coord);
  bool isCoordinatesBased() {return true;}
  void resetCriteria(SegmentSpace*);
  void resetCriteria(TouchSpace*);
  void writeToFile(FILE* data);
  void readFromFile(FILE* data);

 private:
  void setMapping();
  void setUpSlices();
  void computeCutPoints(double* coords1, double* coords2, ShallowArray<double, MAXRETURNRANKS, 100>& cutPoints);
  void addVolumeIndices(double* coords, double radius, ShallowArray<int, MAXRETURNRANKS, 100>& indices);
  int getSliceNumber(double coord, int dim);
  int getVolumeIndex(int sliceIndices[3]);
  int getIndex(int sliceIndices[3]);
  void getVolumeCoords(double* pointCooords, double*& volumeCoords);
    
  int _rank;
  int _numVolumes;             //number of volumes requested
  Tissue* _tissue;
  bool _readHistFromFile;

  int _total;                // total histogram
  double* _columnSizeXYZ;    //the column dimensions
  double* _binwidth;         //width of bins in xyz
  int* _nbinsXYZ;	     //the total number of bins in the histogram for each dimension
  int** _histogramXYZ;       //the histogram values for each dimension
  double* _maxXYZ;
  double* _minXYZ;

  int _nSlicesXYZ[3];	       //the number of slices per dimension (depends on nDims, and nVolumes)
  double *_slicePointsXYZ[3];  //the actual computed slice points for each dimension

  int* _mapping;

  static RunTimeTopology _topology;
};

#endif


