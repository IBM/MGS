// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "VolumeDecomposition.h"
#include "NeuronPartitioner.h"
#include "Communicator.h"
#include "BuffFactor.h"
#include <assert.h>
#include <math.h>
#include <algorithm>

#define ROOT_TWO 1.4142135623730950488016887242097

RunTimeTopology VolumeDecomposition::_topology;

VolumeDecomposition::VolumeDecomposition(
			     int rank,
			     FILE* inputFile,
			     const int numVolumes,
			     Tissue* tissue,
			     int X,
			     int Y,
			     int Z
			     ) :
  _rank(rank),
  _numVolumes(numVolumes),
  _tissue(tissue),
  _readHistFromFile(false),
  _total(0),
  _columnSizeXYZ(0),
  _binwidth(0),
  _nbinsXYZ(0),
  _histogramXYZ(0),
  _maxXYZ(0),
  _minXYZ(0),
  _mapping(0)
{
  if (_topology.runTimeExists()) {
    _nSlicesXYZ[0]=_topology.getX();
    _nSlicesXYZ[1]=_topology.getY();
    _nSlicesXYZ[2]=_topology.getZ();
  }
  else {
    _nSlicesXYZ[0]=X;
    _nSlicesXYZ[1]=Y;
    _nSlicesXYZ[2]=Z;
  }

  if ( (X!=-1 && Y!=-1 && Z!=-1) && (_nSlicesXYZ[0]!=X || _nSlicesXYZ[1]!=Y || _nSlicesXYZ[2]!=Z) ) {
    if (_rank==0)
      std::cerr<<"WARNGING! : VolumeDecomposition : specified dimensions do not match machine topology! ["
	       <<X<<","<<Y<<","<<Z<<"] != ["
	       <<_nSlicesXYZ[0]<<","<<_nSlicesXYZ[1]<<","<<_nSlicesXYZ[2]<<"]"<<std::endl;
  }

  if (_rank==0) printf("MPI Cartesian Dimensions : [%d,%d,%d]\n\n",
		       _nSlicesXYZ[0],_nSlicesXYZ[1],_nSlicesXYZ[2]);

  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size!=_numVolumes) {
    if (_rank==0)
      std::cerr<<"VolumeDecomposition : number of volumes specified does not match number of processors!"<<std::endl;
    MPI_Finalize();
    exit(0);
  }

  setMapping();

  if (inputFile) {
    readFromFile(inputFile);
  }
  else {
    _tissue->generateBins(_columnSizeXYZ, _nbinsXYZ, _binwidth, _maxXYZ, _minXYZ);
    _tissue->generateHistogram(_total, _histogramXYZ);
  }
  setUpSlices();
  decompose();
}

VolumeDecomposition::VolumeDecomposition(VolumeDecomposition& vd) :
  _rank(vd._rank),
  _numVolumes(vd._numVolumes),
  _tissue(vd._tissue),
  _readHistFromFile(vd._readHistFromFile),
  _total(vd._total),
  _columnSizeXYZ(0),
  _nbinsXYZ(0),
  _binwidth(0),
  _histogramXYZ(0),
  _maxXYZ(0),
  _minXYZ(0),
  _mapping(0)
{
  if (_readHistFromFile) {
    _columnSizeXYZ = new double[3]; 
    _nbinsXYZ = new int[3];
    _binwidth = new double[3];
    _histogramXYZ = new int*[3];
    _maxXYZ = new double[3];
    _minXYZ = new double[3];
    for (int d=0; d<3; ++d) {
      _columnSizeXYZ[d] = vd._columnSizeXYZ[d];
      _nbinsXYZ[d] = vd._nbinsXYZ[d];
      _binwidth[d] = vd._binwidth[d];
      _histogramXYZ[d] = new int[_nbinsXYZ[d]];
      for (int dd=0; dd<_nbinsXYZ[d]; ++dd) 
	_histogramXYZ[d][dd] = vd._histogramXYZ[d][dd];
      _maxXYZ[d] = vd._maxXYZ[d];
      _minXYZ[d] = vd._minXYZ[d];
    }
  }
  else {
    _columnSizeXYZ = vd._columnSizeXYZ;
    _nbinsXYZ = vd._nbinsXYZ;
    _binwidth = vd._binwidth;
    _histogramXYZ = vd._histogramXYZ;
    _maxXYZ = vd._maxXYZ;
    _minXYZ = vd._minXYZ;
  }
  for (int d=0; d<3; ++d) {
    _nSlicesXYZ[d] = vd._nSlicesXYZ[d];
    _slicePointsXYZ[d] = new double[_nSlicesXYZ[d]];
    for (int dd=0; dd<_nSlicesXYZ[d]; ++dd)
      _slicePointsXYZ[d][dd]=vd._slicePointsXYZ[d][dd];
  }
  _mapping = new int[_numVolumes];
  for (int i=0; i<_numVolumes; ++i) _mapping[i]=vd._mapping[i];
}

VolumeDecomposition::~VolumeDecomposition()
{
  for (int i=0; i<3; ++i) {
    delete [] _slicePointsXYZ[i];
    if (_readHistFromFile) delete [] _histogramXYZ[i];
  }
  if (_readHistFromFile) {
    delete [] _columnSizeXYZ; 
    delete [] _nbinsXYZ;
    delete [] _binwidth;
    delete [] _histogramXYZ;
    delete [] _maxXYZ;
    delete [] _minXYZ;
  }
  delete [] _mapping;
}

Decomposition* VolumeDecomposition::duplicate()
{
  return new VolumeDecomposition(*this);
}

void VolumeDecomposition::resetCriteria(SegmentSpace* segmentSpace)
{
  _tissue->generateHistogram(_total, _histogramXYZ, segmentSpace, 0);
  decompose();
}

void VolumeDecomposition::resetCriteria(TouchSpace* touchSpace)
{
  _tissue->generateHistogram(_total, _histogramXYZ, 0, touchSpace);
  decompose();
}

void VolumeDecomposition::writeToFile(FILE* data)
{
  fwrite(&_nSlicesXYZ, sizeof(int), 3, data);
  fwrite(&_total, sizeof(int), 1, data);
  fwrite(_columnSizeXYZ, sizeof(double), 3, data);
  fwrite(_nbinsXYZ, sizeof(int), 3, data);
  fwrite(_binwidth, sizeof(double), 3, data);
  for (int d=0; d<3; ++d) {
    fwrite(_histogramXYZ[d], sizeof(int), _nbinsXYZ[d], data);
  }
  fwrite(_maxXYZ, sizeof(double), 3, data);
  fwrite(_minXYZ, sizeof(double), 3, data);
}

void VolumeDecomposition::readFromFile(FILE* data)
{
  _readHistFromFile=true;

  int nSlices[3];
  size_t s=fread(&nSlices, sizeof(int), 3, data);
  if ( _nSlicesXYZ[0]!=nSlices[0] || _nSlicesXYZ[1]!=nSlices[1] || _nSlicesXYZ[2]!=nSlices[2] ) {
    if (_rank==0)
      std::cerr<<"VolumeDecomposition : volume dimensions in binary file and tissue specification do not match! ["
	       <<nSlices[0]<<","<<nSlices[1]<<","<<nSlices[2]<<"] != ["
	       <<_nSlicesXYZ[0]<<","<<_nSlicesXYZ[1]<<","<<_nSlicesXYZ[2]<<"]"<<std::endl;
    MPI_Finalize();
    exit(0);
  }

  s=fread(&_total, sizeof(int), 1, data);		//total number of points in column
  _columnSizeXYZ = new double[3];
  s=fread(_columnSizeXYZ, sizeof(double), 3, data);
  _nbinsXYZ = new int[3];
  s=fread(_nbinsXYZ, sizeof(int), 3, data);
  _binwidth = new double[3];
  s=fread(_binwidth, sizeof(double), 3, data);
  _histogramXYZ=new int*[3];
  for (int d=0; d<3; ++d) {
    _histogramXYZ[d] = new int[_nbinsXYZ[d]];
    s=fread(_histogramXYZ[d], sizeof(int), _nbinsXYZ[d], data);
  }
  _maxXYZ = new double[3];
  _minXYZ = new double[3];
  s=fread(_maxXYZ, sizeof(double), 3, data);
  s=fread(_minXYZ, sizeof(double), 3, data);
}


void VolumeDecomposition::decompose()
{
  bool exitCondition=false;
  if (_nSlicesXYZ[0]>_nbinsXYZ[0]) {
    std::cerr<<"The number of slices specified for dimension 0 ("<<_nSlicesXYZ[0]<<") exceeds the number\n"
	     <<"of bins for dimension 0 ("<<_nbinsXYZ[0]<<"). " 
	     <<"Consider reducing the number of slices."<<std::endl;
    exitCondition=true;
  }
  if (_nSlicesXYZ[1]>_nbinsXYZ[1]) {
    std::cerr<<"The number of slices specified for dimension 1 ("<<_nSlicesXYZ[1]<<") exceeds the number\n"
	     <<"of bins for dimension 1 ("<<_nbinsXYZ[1]<<"). "
	     <<"Consider reducing the number of slices."<<std::endl;
    exitCondition=true;
  }
  if (_nSlicesXYZ[2]>_nbinsXYZ[2]) {
    std::cerr<<"The number of slices specified for dimension 2 ("<<_nSlicesXYZ[2]<<") exceeds the number\n"
	     <<"of bins for dimension 2 ("<<_nbinsXYZ[2]<<"). "
	     <<"Consider reducing the number of slices."<<std::endl;    exitCondition=true;
  }   	
  if (exitCondition) exit(-1);
 
  for(int d=0; d<3; ++d) {//the three dimensions  
    double segsPerSlice = double(_total)/double(_nSlicesXYZ[d]); // number of segments per slice
    int last=-1;
    double segs = 0;
    int j=0;
    for(int i=0; i<_nSlicesXYZ[d]; ++i) { // nSlicesXYZ is number of slices per dimension
      while (j<_nbinsXYZ[d]) {
        if (segs>=segsPerSlice) {
	  double overrun = segs-segsPerSlice;
	  double binFraction = overrun/(double)(_histogramXYZ[d][j-1]);      
	  _slicePointsXYZ[d][i] = _minXYZ[d] + _binwidth[d] * ((double)j-binFraction);
	  last = i;
	  segs = overrun;
	  break;
        }
        segs += double(_histogramXYZ[d][j]);
        ++j;
      }
    }
    if (last+1<_nSlicesXYZ[d]) ++last;
    assert(last==_nSlicesXYZ[d]-1);
    _slicePointsXYZ[d][last] = _maxXYZ[d]+_binwidth[d];
  }
}

void VolumeDecomposition::setMapping()
{
  delete [] _mapping;
  _mapping = new int[_numVolumes];

  // Set up coordinate to ranks mapping
  MPI_Comm cart_comm;
  int periods[3] = {0,0,0};
  int reorder = 0;
  MPI_Cart_create(MPI_COMM_WORLD, 3, _nSlicesXYZ, periods, reorder, &cart_comm);

  int rank=0, coords[3];
  MPI_Group gc, gw;
  int rc, rw;
  MPI_Comm_group(cart_comm,&gc);
  MPI_Comm_group(MPI_COMM_WORLD,&gw);

  for (coords[0]=0; coords[0]<_nSlicesXYZ[0]; coords[0]+=1) {
    for (coords[1]=0; coords[1]<_nSlicesXYZ[1]; coords[1]+=1) {
      for (coords[2]=0; coords[2]<_nSlicesXYZ[2]; coords[2]+=1) {
	MPI_Cart_rank(cart_comm,coords,&rc);
	MPI_Group_translate_ranks(gc,1,&rc,gw,&rw);
	assert(rw!=MPI_UNDEFINED);
	_mapping[getIndex(coords)]=rw;
      }
    }
  }
  MPI_Comm_free(&cart_comm);
}

void VolumeDecomposition::getRanks(Sphere* sphere, double* coords2, double deltaRadius, ShallowArray<int, MAXRETURNRANKS, 100>& ranks)
{
  ranks.clear();
  addRanks(sphere, coords2, deltaRadius, ranks);
  ranks.sort();
  ranks.unique();
}

bool VolumeDecomposition::mapsToRank(double* coords, double radius, int rank)
{
  bool rval=false;
  int sliceIndices[3];
  int lo[3], hi[3];
  for (int d=0; d<3; ++d) {
    lo[d] = getSliceNumber(coords[d]-radius, d);
    hi[d] = getSliceNumber(coords[d]+radius, d);
  }
  for (sliceIndices[0]=lo[0]; !rval && sliceIndices[0]<=hi[0]; ++sliceIndices[0]) {
    for (sliceIndices[1]=lo[1]; !rval && sliceIndices[1]<=hi[1]; ++sliceIndices[1]) {
      for (sliceIndices[2]=lo[2]; !rval && sliceIndices[2]<=hi[2]; ++sliceIndices[2]) {
	if (getVolumeIndex(sliceIndices)==rank) rval=true;
      }
    }
  }
  return rval;
}

bool VolumeDecomposition::mapsToRank(Sphere* sphere, double* coords2, double deltaRadius, int rank)
{
  bool rval=false;
  double* coords = sphere->_coords;
  double radius=sphere->_radius+deltaRadius;
  rval=mapsToRank(coords, radius, rank);

  if (!rval) rval = mapsToRank(coords2, radius, rank);
  
  if (!rval) {
    ShallowArray<double, MAXRETURNRANKS, 100> cutPoints;
    computeCutPoints(sphere->_coords, coords2, cutPoints);
    assert(cutPoints.size()%3==0);
    for (int i=0; !rval && i<cutPoints.size(); i+=3) {
      double coords[3]={cutPoints[i], cutPoints[i+1], cutPoints[i+2]};
      rval = mapsToRank(coords, radius, rank);					
    }
  }
  return rval;
}

void VolumeDecomposition::getRanks(Sphere* sphere, double deltaRadius, ShallowArray<int, MAXRETURNRANKS, 100>& ranks)
{
  ranks.clear();
  addVolumeIndices(sphere->_coords, sphere->_radius+deltaRadius, ranks);
  ranks.sort();
  ranks.unique();
}

void VolumeDecomposition::computeCutPoints(double* coords1, double* coords2, ShallowArray<double, MAXRETURNRANKS, 100>& cutPoints)
{
  double diff[3];
  for (int d=0; d<3; ++d) {
    diff[d] = coords2[d]-coords1[d]; // dx, dy, dz
  }

  double deltas[3][3];	
  for (int i=0; i<3; ++i) {
    for (int j=0; j<3; ++j) {
      deltas[i][j] = diff[i]/diff[j];
    }
  }

  int first[3];
  int last[3];
  int numCutPoints = 0;
  for (int d=0; d<3; ++d) {
    first[d] = getSliceNumber(coords1[d], d);
    last[d] = getSliceNumber(coords2[d], d);
    numCutPoints += abs(last[d]-first[d]);
  }  
  assert(numCutPoints<=MAXRETURNRANKS);

  for (int i=0; i<3; ++i) {
    int next, begin, end;
    if (last[i]>=first[i]) {
      next = 1;
      begin = first[i];
      end = last[i];
    }
    else {
      next = -1;
      begin = first[i]-1;
      end = last[i]-1;
    }
    for (int sliceNumber=begin; sliceNumber!=end; sliceNumber+=next) {
      double delta = _slicePointsXYZ[i][sliceNumber] - coords1[i];
      for (int j=0; j<3; ++j) {
	cutPoints.push_back(coords1[j] + deltas[j][i] * delta);
      } 
    }
  }	
}

void VolumeDecomposition::addRanks(Sphere* sphere, double* coords2, double deltaRadius, ShallowArray<int, MAXRETURNRANKS, 100>& ranks)
{
  double radius=sphere->_radius+deltaRadius;
  addVolumeIndices(sphere->_coords, radius, ranks);
  addVolumeIndices(coords2, radius, ranks);
  ShallowArray<double, MAXRETURNRANKS, 100> cutPoints;
  computeCutPoints(sphere->_coords, coords2, cutPoints);
  assert(cutPoints.size()%3==0);
  for (int i=0; i<cutPoints.size(); i+=3) {
    double coords[3]={cutPoints[i], cutPoints[i+1], cutPoints[i+2]};
    addVolumeIndices(coords, radius, ranks);					
  }
}

void VolumeDecomposition::addVolumeIndices(double* coords, double radius, ShallowArray<int, MAXRETURNRANKS, 100>& indices)
{

  int sliceIndices[3];
  int lo[3], hi[3];
  for (int d=0; d<3; ++d) {
    lo[d] = getSliceNumber(coords[d]-radius, d);
    hi[d] = getSliceNumber(coords[d]+radius, d);
  }
  for (sliceIndices[0]=lo[0]; sliceIndices[0]<=hi[0]; ++sliceIndices[0]) {
    for (sliceIndices[1]=lo[1]; sliceIndices[1]<=hi[1]; ++sliceIndices[1]) {
      for (sliceIndices[2]=lo[2]; sliceIndices[2]<=hi[2]; ++sliceIndices[2]) {
	indices.push_back(getVolumeIndex(sliceIndices));
      }
    }
  }
}

int VolumeDecomposition::getSliceNumber(double coord, int dim)
{
  //Binary search
  int lo = 0, hi = _nSlicesXYZ[dim]-1, mid=0;
  double* slicePointsXYZ = _slicePointsXYZ[dim];

  while(lo < hi) {  
    mid = (int)(double(lo + hi) * 0.5);   
    if(coord <= slicePointsXYZ[mid]) 
      hi = mid;
    else
      lo = mid+1;
  }
  return lo;
}

//GOAL: return the rank of the MPI process at which the 
//  coordinate of the enter of the Sphere 'sphere'
//  falls within
int VolumeDecomposition::getRank(Sphere& sphere)
{
  double* coords=sphere._coords;
  int sliceIndices[3];
  sliceIndices[0] = getSliceNumber(coords[0], 0);
  sliceIndices[1] = getSliceNumber(coords[1], 1);
  sliceIndices[2] = getSliceNumber(coords[2], 2);
  return getVolumeIndex(sliceIndices);	
}

int VolumeDecomposition::getRank(double* coords)
{
  int sliceIndices[3];
  sliceIndices[0] = getSliceNumber(coords[0], 0);
  sliceIndices[1] = getSliceNumber(coords[1], 1);
  sliceIndices[2] = getSliceNumber(coords[2], 2);
  return getVolumeIndex(sliceIndices);	
}

int VolumeDecomposition::getVolumeIndex(int sliceIndices[3] )
{
  return _mapping[getIndex(sliceIndices)];
}

int VolumeDecomposition::getIndex(int sliceIndices[3])
{  
  return (sliceIndices[0]*_nSlicesXYZ[1]+sliceIndices[1])*_nSlicesXYZ[2]+sliceIndices[2];
}

void VolumeDecomposition::getVolumeCoords(double* pointCoords, double*& volumeCoords)
{
  volumeCoords[0] = getSliceNumber(pointCoords[0], 0);
  volumeCoords[1] = getSliceNumber(pointCoords[1], 1);
  volumeCoords[2] = getSliceNumber(pointCoords[2], 2);
}

void VolumeDecomposition::setUpSlices()
{
  assert(_numVolumes);
  if (_nSlicesXYZ[0]<=0 || _nSlicesXYZ[1]<=0 || _nSlicesXYZ[2]<=0) {
    int rt = int(round(cbrt(double(_numVolumes))));
    if (_numVolumes != rt*rt*rt) {
      std::cerr<<"Error: nTouchDetectors ("<<_numVolumes<<") must be an integer cubed for 3D slicing : "	     
	       <<rt<<" != "<<cbrt(double(_numVolumes))<<std::endl;
      exit(0);
    }
    _nSlicesXYZ[0] = rt;
    _nSlicesXYZ[1] = rt;
    _nSlicesXYZ[2] = rt;
  }
  else if (_nSlicesXYZ[0]*_nSlicesXYZ[1]*_nSlicesXYZ[2]!=_numVolumes) {
    std::cerr<<"Slicing geometry does not match number of machine nodes!"<<std::endl;
    exit(0);
  }
  for(int d=0; d<3; ++d) {//the three dimensions  
    _slicePointsXYZ[d] = new double[_nSlicesXYZ[d]];
  }
}
