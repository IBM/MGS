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

#include "SegmentForceAggregator.h"
#include "SegmentForce.h"
#include "Tissue.h"
#include "Branch.h"
#include "Segment.h"
#include "BuffFactor.h"

#include <cassert>
#include <algorithm>
#include <list>

SegmentForceAggregator::SegmentForceAggregator(
					       int rank,
					       int nSlicers,
					       int nSegmentForceDetectors,
					       Tissue* tissue)
  : _rank(rank),
    _nSlicers(nSlicers),
    _nSegmentForceDetectors(nSegmentForceDetectors),
    _numberOfSenders(0),
    _tissue(tissue),
    _segmentForcesPerSender(0),
    _numberOfInts(1),
    _typeInt(MPI_INT),
    _segmentForces(0),
    _segmentForceDispls(0),
    _segmentForceDataSize(0),
    _previousSegmentForceDataSize(1),
    _E(0)
{
  _numberOfSenders = (nSlicers>nSegmentForceDetectors)?nSlicers:nSegmentForceDetectors;

  MPI_Type_contiguous(N_SEGFORCE_DATA, MPI_DOUBLE, &_typeSegmentForceData);       
  MPI_Type_commit(&_typeSegmentForceData);
  
  _segmentForcesPerSender = new int[_numberOfSenders];
  _segmentForceDispls = new int[_numberOfSenders];
  for (int i=0; i<_numberOfSenders; ++i) {
    _segmentForcesPerSender[i] = 0;
    _segmentForceDispls[i] = 0;
  }
  _segmentForces = new double[_previousSegmentForceDataSize*N_SEGFORCE_DATA];
}

double SegmentForceAggregator::aggregate(int frontNumber)
{
  double* forceEnd = _segmentForces + _segmentForceDataSize*N_SEGFORCE_DATA;
  int neuronIndex, branchIndex, segmentIndex;
  Neuron* neurons = _tissue->getNeurons();
  for (double* segForce = _segmentForces; segForce!=forceEnd; segForce+=N_SEGFORCE_DATA) {
    double s1Key=segForce[0];
    neuronIndex= _tissue->getNeuronIndex(_segmentDescriptor.getNeuronIndex(s1Key));
    branchIndex=_segmentDescriptor.getBranchIndex(s1Key);
    segmentIndex=_segmentDescriptor.getSegmentIndex(s1Key);
    assert(segmentIndex!=0);
    Segment& s = neurons[neuronIndex].getBranches()[branchIndex].getSegments()[segmentIndex]; 
    //assert(frontNumber==s.getFrontLevel());    
    s.addForce(&segForce[1]);
  }
  return _E;
}

void SegmentForceAggregator::prepareToReceive(int receiveCycle, int receivePhase, CommunicatorFunction& funPtrRef)
{
  assert(receiveCycle==0);
  switch (receivePhase) {
  case 0 :
    funPtrRef = &Communicator::allToAll; 
    break;
  case 1 :
    initializePhase1Receive();
    funPtrRef = &Communicator::allToAllV; 
    break;
  case 2 :
    funPtrRef = &Communicator::allReduceSum; 
    break;
  default : assert(0);
  }
}

void* SegmentForceAggregator::getRecvbuf(int receiveCycle, int receivePhase)
{
  assert(receiveCycle==0);
  void* rval;
  switch (receivePhase) {
  case 0 : rval = (void*)_segmentForcesPerSender; break;
  case 1 : rval = (void*)_segmentForces; break;
  case 2 : rval = (void*)&_E; break;
  default : assert(0);
  }
  return rval;
}

int* SegmentForceAggregator::getRecvcounts(int receiveCycle, int receivePhase)
{
  assert(receiveCycle==0);
  int* rval;
  switch (receivePhase) {
  case 0 : rval = &_numberOfInts; break;
  case 1 : rval = _segmentForcesPerSender; break;
  default : assert(0);
  }
  return rval;
}

int* SegmentForceAggregator::getRdispls(int receiveCycle, int receivePhase)
{
  assert(receiveCycle==0);
  int* rval;
  switch (receivePhase) {
  case 0 : assert(0); break;
  case 1 : rval = _segmentForceDispls; break;
  default : assert(0);
  }
  return rval;
}

MPI_Datatype* SegmentForceAggregator::getRecvtypes(int receiveCycle, int receivePhase)
{
  assert(receiveCycle==0);
  MPI_Datatype* rval;
  switch (receivePhase) {
  case 0 : rval = &_typeInt; break;
  case 1 : rval = &_typeSegmentForceData; break;
  default : assert(0);
  }
  return rval;
}

void SegmentForceAggregator::initializePhase1Receive()
{
  _segmentForceDataSize = 0;
  for(int i=0; i<_numberOfSenders; i++) {
    _segmentForceDispls[i] = _segmentForceDataSize;
    _segmentForceDataSize += _segmentForcesPerSender[i];
  }
  Segment *segmentPtr = _tissue->getSegments(),
    *segmentsEnd = segmentPtr+_tissue->getSegmentArraySize();
  if (_segmentForceDataSize>_previousSegmentForceDataSize) {
    delete [] _segmentForces;
    _segmentForces = new double[getBuffAllocationSize(_segmentForceDataSize)*N_SEGFORCE_DATA];
    _previousSegmentForceDataSize =  getUsableBuffSize(_segmentForceDataSize);
  }
}

int SegmentForceAggregator::getNumberOfReceivePhasesPerCycle(int receiveCycle)
{
  assert(receiveCycle==0);
  return SEGMENTFORCEAGGREGATOR_RECEIVE_PHASES;
}

SegmentForceAggregator::~SegmentForceAggregator()
{
  delete [] _segmentForcesPerSender;
  delete [] _segmentForceDispls;
  delete [] _segmentForces;
}
