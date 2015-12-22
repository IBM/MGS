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

#include "TissueSlicer.h"
#include "Tissue.h"
#include "Neuron.h"
#include "Branch.h"
#include "Segment.h"
#include "Decomposition.h"
#include "Capsule.h"
#include "VecPrim.h"
#include "Params.h"
#include "ComputeBranch.h"
#include "BuffFactor.h"
#include <math.h>
#include <cassert>
#include <list>
#include <algorithm>


TissueSlicer::TissueSlicer(
			   const int rank,
			   const int nSlicers,
			   const int nTouchDetectors,
			   Tissue* tissue,
			   Decomposition** decomposition,
			   Params* params) :
  _rank(rank),
  _nSlicers(nSlicers),
  _nTouchDetectors(nTouchDetectors),
  _tissue(tissue),
  _decomposition(decomposition),
  _sliceSegmentIndices(0),
  _numberOfReceivers(0),
  _maxNumSegs(1),
  _segmentBlockLengths(0), 
  _segmentBlockDisplacements(0), 
  _segmentsPerSlice(0),
  _numberOfInts(1),
  _typeInt(MPI_INT),
  _segmentCounts(0),
  _segmentDispls(0),
  _typeSegments(0),
  _typeSegmentData(0),
  _sendBuff(0),
  _sendBuffSize(1),
  _dataSize(0),
  _params(0)
{
  if (params) {
    _params = new Params(*params);
  }

  _numberOfReceivers = (_nSlicers>_nTouchDetectors)?_nSlicers:_nTouchDetectors;
  assert(_numberOfReceivers>0);

#ifdef A2AW
  _typeSegments = new MPI_Datatype[_numberOfReceivers];
  _segmentBlockLengths = new int[_maxNumSegs];
  _segmentBlockDisplacements = new int[_maxNumSegs];
  for (int j=0; j<_maxNumSegs; ++j) {
    _segmentBlockLengths[j] = 1;
    _segmentBlockDisplacements[j] = 0;
  }
#else
  _typeSegments = new MPI_Datatype[1];
  _sendBuff = new double[_sendBuffSize];
#endif

  _sliceSegmentIndices = new std::vector<long int>[_numberOfReceivers];
  _segmentCounts = new int[_numberOfReceivers];
  _segmentDispls = new int[_numberOfReceivers];
  _segmentsPerSlice = new int[_numberOfReceivers];
  for (int i=0; i<_numberOfReceivers; ++i) {
    _segmentCounts[i] = 1;
    _segmentDispls[i] = 0;
    _segmentsPerSlice[i] = 0;
  }
}

TissueSlicer::~TissueSlicer()
{
  delete [] _segmentCounts;
  delete [] _segmentDispls;
  delete [] _typeSegments;
  delete [] _segmentsPerSlice;
  delete [] _sliceSegmentIndices;
  delete [] _segmentBlockLengths;
  delete [] _segmentBlockDisplacements;
  delete [] _sendBuff;
  delete _params;
}

void TissueSlicer::initializePhase0()
{
  assert(_dataSize>0);
#ifdef A2AW
  for (int i=0; i<_numberOfReceivers; ++i) {
    _segmentsPerSlice[i] = _sliceSegmentIndices[i].size();
  }
#else
  int sendBuffSize=0;
  for (int i=0; i<_numberOfReceivers; ++i) {
    _segmentDispls[i] = sendBuffSize;
    sendBuffSize += (_segmentsPerSlice[i] = _sliceSegmentIndices[i].size());
  }
  sendBuffSize*=_dataSize;
  if (sendBuffSize>_sendBuffSize) {
    delete [] _sendBuff;
    _sendBuff = new double[getBuffAllocationSize(sendBuffSize)];
    _sendBuffSize=getUsableBuffSize(sendBuffSize);
  }

#endif
}

void TissueSlicer::initializePhase1()
{
#ifdef A2AW
  int currentMaxNumSegs=0;
  for (int i=0; i<_numberOfReceivers; ++i) {
    int numSegs = _segmentsPerSlice[i];
    if (numSegs>currentMaxNumSegs) currentMaxNumSegs = numSegs;
  }

  if (_maxNumSegs < currentMaxNumSegs) {
    _maxNumSegs = getBuffSize(currentMaxNumSegs);
    delete [] _segmentBlockLengths;
    delete [] _segmentBlockDisplacements;
    _segmentBlockLengths = new int[_maxNumSegs];
    _segmentBlockDisplacements = new int[_maxNumSegs];
    for (int j=0; j<_maxNumSegs; ++j) {
      _segmentBlockLengths[j] = 1;
      _segmentBlockDisplacements[j] = 0;
    }
  }

  for (int i=0; i<_numberOfReceivers; ++i) {
    int numSegs = _segmentsPerSlice[i];
    for (int j=0; j<numSegs; ++j)  _segmentBlockDisplacements[j] = (_sliceSegmentIndices[i])[j];
    MPI_Type_free(&_typeSegments[i]);
    MPI_Type_indexed(numSegs, _segmentBlockLengths, _segmentBlockDisplacements, _typeSegmentData, &_typeSegments[i]);
    MPI_Type_commit(&_typeSegments[i]);
  }
#else
  int writePos=0;
  for (int i=0; i<_numberOfReceivers; ++i) {
    int numSegs = _segmentsPerSlice[i];
    for (int j=0; j<numSegs; ++j) {
      writeBuff(i, j, writePos);
    }
  }
#endif
}

void TissueSlicer::prepareToSend(int sendCycle, int sendPhase, CommunicatorFunction& funPtrRef)
{  
  assert(sendCycle==0);
  switch (sendPhase) {
  case 0 :
    sliceAllNeurons();
    initializePhase0();
    funPtrRef = &Communicator::allToAll; 
    break;
  case 1 :
    initializePhase1();
#ifdef A2AW
    funPtrRef = &Communicator::allToAllW;
#else
    funPtrRef = &Communicator::allToAllV;
#endif
    break;
  default : assert(0);
  }
}


void* TissueSlicer::getSendbuf(int sendCycle, int sendPhase)
{  
  assert(sendCycle==0);
  void* rval;
  switch (sendPhase) {
  case 0 : rval = (void*)_segmentsPerSlice; break;
  case 1 : rval = getSendBuff(); break;
  default : assert(0);
  }
  return rval;
}

int* TissueSlicer::getSendcounts(int sendCycle, int sendPhase)
{
  int* rval;
  switch (sendPhase) {
  case 0 : rval = &_numberOfInts; break;
#ifdef A2AW
  case 1 : rval = _segmentCounts;
#else
  case 1 : rval = _segmentsPerSlice;
#endif
    break;
  default : assert(0);
  }
  return rval;
}
  
int* TissueSlicer::getSdispls(int sendCycle, int sendPhase)
{
  assert(sendCycle==0);
  int* rval;
  switch (sendPhase) {
  case 0 : assert(0); break;
  case 1 : rval = _segmentDispls; break;
  default : assert(0);
  }
  return rval;
}

MPI_Datatype* TissueSlicer::getSendtypes(int sendCycle, int sendPhase)
{
  assert(sendCycle==0);
  MPI_Datatype* rval;
  switch (sendPhase) {
  case 0 : rval = &_typeInt; break;
  case 1 : rval = _typeSegments; break;
  default : assert(0);
  }
  return rval;
}

int TissueSlicer::getNumberOfSendPhasesPerCycle(int sendCycle)
{
  assert(sendCycle==0);
  return TISSUE_SLICER_SEND_PHASES;
}
