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

#include "NeuroDevTissueSlicer.h"
#include "Tissue.h"
#include "Neuron.h"
#include "Branch.h"
#include "Segment.h"
#include "Decomposition.h"
#include "SegmentSpace.h"
#include "Params.h"
#include "VecPrim.h"
#include "ShallowArray.h"
#include <math.h>
#include <cassert>


#define WRITE_SZ1 (N_SEG_DATA-3)
#define WRITE_SZ2 3

#define ANTI_DISTORTION_CONSTANT_INDEX 4

NeuroDevTissueSlicer::NeuroDevTissueSlicer(
			   const int rank,
			   const int nSlicers,
			   const int nTouchDetectors,
			   Tissue* tissue,
			   Decomposition** decomposition,
			   SegmentSpace* segmentSpace,
			   Params* params,
			   double& E) :
  TissueSlicer(rank,
	       nSlicers,
	       nTouchDetectors,
	       tissue,
	       decomposition,
	       params),  
  _segmentSpace(segmentSpace),
  _E(E),
  _segs(0),
  _segsEnd(0),
  _segsSize(0),
  _forces(0),
  _forcesEnd(0),
  _forcesSize(0),
  _R0a(0),
  _R0b(0)

{
  if (_tissue->isEmpty()) {
    _segsSize = 1;
    _segs = new Segment[_segsSize];
    _R0a = new double[_segsSize];
    _R0b = new double[_segsSize];
    _forcesSize = 3;
    _forces = new double[_forcesSize];
  }
  else {
    _segs = _tissue->getSegments();
    _segsSize = _tissue->getSegmentArraySize();
    _R0a = new double[_segsSize];
    _R0b = new double[_segsSize];
    _forcesSize = 3*_segsSize;
    _forces = new double[_forcesSize];
  }
  _segsEnd = _segs+_segsSize;
  _forcesEnd = _forces + _forcesSize;
  _dataSize = N_SEG_DATA;

#ifdef A2AW
  int blocklen[2] = {WRITE_SZ1, WRITE_SZ2};

  MPI_Aint segAddress;
  MPI_Aint disp[2];

  MPI_Get_address(&_segs[0], &segAddress);
  MPI_Get_address(_segs[0].getCoords(), &disp[0]);
  MPI_Get_address(&_forces[0], &disp[1]);

  disp[0] -= segAddress;
  disp[1] -= segAddress;

  MPI_Datatype typeSegmentDataBasic;
  MPI_Type_create_hindexed(2, blocklen, disp, MPI_DOUBLE, &typeSegmentDataBasic);
  MPI_Type_create_resized(typeSegmentDataBasic, 0,
			  sizeof(Segment), &_typeSegmentData);
  MPI_Type_commit(&_typeSegmentData);

  for (int i=0; i<_numberOfReceivers; ++i) {
    int numSegs = 1;
    MPI_Type_indexed(numSegs, _segmentBlockLengths, _segmentBlockDisplacements, _typeSegmentData, &_typeSegments[i]);
    MPI_Type_commit(&_typeSegments[i]);
  }
#else
  MPI_Type_contiguous(_dataSize, MPI_DOUBLE, &_typeSegments[0]);
  MPI_Type_commit(&_typeSegments[0]);
#endif
  for (int i=0; i<_segsSize; ++i) {
    if (_segs[i].getSegmentIndex()==0) _R0a[i]=0;
    else _R0a[i]=sqrt(SqDist(_segs[i].getOrigCoords(), 
			    _segs[i-1].getOrigCoords())) *
      _params->getBondR0(_segmentDescriptor.getBranchType(_segs[i].getSegmentKey()));
    _R0b[i]=-1.0;
  }
}

NeuroDevTissueSlicer::~NeuroDevTissueSlicer()
{
  if (_tissue->isEmpty()) delete [] _segs;
  delete [] _R0a;
  delete [] _R0b;
  delete [] _forces;
}

void NeuroDevTissueSlicer::sliceAllNeurons()
{
  if (!_tissue->isEmpty()) {
    Decomposition* decomposition = *_decomposition;
    for (int i=0; i<_numberOfReceivers; ++i) _sliceSegmentIndices[i].clear();
    Segment* seg = _segs;
    ShallowArray<int, MAXRETURNRANKS, 100> indices;
    ShallowArray<int, MAXRETURNRANKS, 100>::iterator iter, end;	  
    _E=0;
    for (int idx=0; seg < _segsEnd; ++seg, ++idx) {
      // eliminate first segment of branches, as these are redundant for the purposes of neuroDev
      if (_segmentSpace->isInSpace(seg) && seg->getSegmentIndex()>0 ) {
	computeTopologicalForces(idx);
	decomposition->getRanks(&seg->getSphere(),
				_params ? _params->getRadius(seg->getSegmentKey()) : 0,
				indices);
	end = indices.end();
	for (iter=indices.begin(); iter!=end; ++iter) {
	  _sliceSegmentIndices[(*iter)].push_back(idx);
	}
      }
    }
  }
}

void NeuroDevTissueSlicer::computeTopologicalForces(int idx)
{

  Segment* seg = &_segs[idx];
  double key = seg->getSegmentKey();
  int typeA = _segmentDescriptor.getBranchType(key);
  double* Fa = &_forces[3*idx];
  memset(Fa, 0, 3*sizeof(double));

  double lengthK0=_params->getBondK0(typeA);
  double distortionK0=_params->getBondK0(ANTI_DISTORTION_CONSTANT_INDEX);
  
  if (lengthK0 != 0.0 || distortionK0 != 0.0) {
    
    double *cds = seg->getCoords();
    double* prevCds = _segs[idx-1].getCoords();
    double F[3];
    double E=0;
    
    ////
    // Compute Segment Length Term
    ////
    
    if (lengthK0 != 0.0) {
      _segmentForce.HarmonicDistance(prevCds,
				     cds,
				     _R0a[idx],
				     lengthK0,
				     E,
				     F);
      
      _E += E;
      for (int ii=0; ii<3; ++ii) Fa[ii] += F[ii];
    }

    ////
    // Compute Anti-Distortion Term
    ////

    if (distortionK0 != 0.0) {
      double *cdsCB = seg->getBranch()->getNeuron()->getSegmentsBegin()->getCoords();
      double *cdsT1=seg->getBranch()->getTerminalSegment()->getCoords();
      double *cdsT2=seg->getBranch()->getDisplacedTerminalCoords();
      if (_R0b[idx]<0) _R0b[idx]=sqrt(SqDist(prevCds, cdsT2))-_R0a[idx];
      _segmentForce.HarmonicDistance(cdsT2,
				     cds,
				     _R0b[idx],
				     distortionK0,
				     E,
				     F);
      _E += E;
      for (int ii=0; ii<3; ++ii) Fa[ii] += F[ii];
    }
  }
}

void NeuroDevTissueSlicer::writeBuff(int i, int j, int& writePos)
{
  assert(writePos<=_sendBuffSize-(WRITE_SZ1+WRITE_SZ2));
  memcpy(&_sendBuff[writePos], _segs[(_sliceSegmentIndices[i])[j]].getCoords(), WRITE_SZ1*sizeof(double));
  writePos += WRITE_SZ1;
  memcpy(&_sendBuff[writePos], &_forces[3*(_sliceSegmentIndices[i])[j]], WRITE_SZ2*sizeof(double));
  writePos += WRITE_SZ2;
}

void* NeuroDevTissueSlicer::getSendBuff()
{
#ifdef A2AW
  return (void*)_segs;
#else
  return (void*)_sendBuff;
#endif
}
