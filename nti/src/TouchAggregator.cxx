// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-07-18-2017
//
// (C) Copyright IBM Corp. and EPFL 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// ================================================================

#include "TouchAggregator.h"
#include "Touch.h"
#include "Tissue.h"
#include "Branch.h"
#include "Segment.h"
#include "TissueContext.h"
#include "TouchVector.h"
#include "Block.h"
#include "BuffFactor.h"

#include <cassert>
#include <algorithm>
#include <list>

TouchAggregator::TouchAggregator(
			   int rank,
			   int nSenders,
			   TissueContext* tissueContext)
  : _rank(rank),
    _numberOfSenders(nSenders),
    _touchesPerSender(0),
    _numberOfInts(1),
    _typeInt(MPI_INT),
    _touches(0),
    _recvbuf(0),
    _touchDispls(0),
    _typeTouches(0),
    _touchDataSize(0),
    _previousTouchDataSize(1),
    _tissueContext(tissueContext)
{
#ifdef A2AW
  _typeTouches = new MPI_Datatype[_numberOfSenders]; 
#else
  _typeTouches = new MPI_Datatype[1];
  _typeTouches[0]=*(Touch::getTypeTouch());
#endif
  _touchesPerSender = new int[_numberOfSenders];
  _touchDispls = new int[_numberOfSenders];
  for (int i=0; i<_numberOfSenders; ++i) {
    _touchesPerSender[i] = 0;
    _touchDispls[i] = 0;
#ifdef A2AW
    _typeTouches[i]=*(Touch::getTypeTouch());
#endif
  }
  _touches = new Touch[_previousTouchDataSize];
}

void TouchAggregator::prepareToReceive(int receiveCycle, int receivePhase, CommunicatorFunction& funPtrRef)
{
  assert(receiveCycle==0);
  switch (receivePhase) {
  case 0 :
    funPtrRef = &Communicator::allToAll;
    break;
  case 1 :
    initializePhase1Receive();
#ifdef A2AW
    funPtrRef = &Communicator::allToAllW; 
#else
    funPtrRef = &Communicator::allToAllV; 
#endif
    break;
  default : assert(0);
  }
}

void* TouchAggregator::getRecvbuf(int receiveCycle, int receivePhase)
{
  assert(receiveCycle==0);
  void* rval;
  switch (receivePhase) {
  case 0 : rval = (void*)_touchesPerSender; break;
  case 1 : rval = (void*)_recvbuf; break;
  default : assert(0);
  }
  return rval;
}

int* TouchAggregator::getRecvcounts(int receiveCycle, int receivePhase)
{
  assert(receiveCycle==0);
  int* rval;
  switch (receivePhase) {
  case 0 : rval = &_numberOfInts; break;
  case 1 : rval = _touchesPerSender; break;
  default : assert(0);
  }
  return rval;
}

int* TouchAggregator::getRdispls(int receiveCycle, int receivePhase)
{
  assert(receiveCycle==0);
  int* rval;
  switch (receivePhase) {
  case 0 : assert(0); break;
  case 1 : rval = _touchDispls; break;
  default : assert(0);
  }
  return rval;
}

MPI_Datatype* TouchAggregator::getRecvtypes(int receiveCycle, int receivePhase)
{
  assert(receiveCycle==0);
  MPI_Datatype* rval;
  switch (receivePhase) {
  case 0 : rval = &_typeInt; break;
  case 1 : rval = _typeTouches; break;
  default : assert(0);
  }
  return rval;
}

void TouchAggregator::initializePhase1Receive()
{
  _touchDataSize = 0;
  for(int i=0; i<_numberOfSenders; i++) {
#ifdef A2AW
    _touchDispls[i] = _touchDataSize*sizeof(Touch);
#else
    _touchDispls[i] = _touchDataSize;
#endif
    _touchDataSize += _touchesPerSender[i];
  }
  if (_touchDataSize>_previousTouchDataSize || _tissueContext) {
    if (_tissueContext) {
      Block<Touch>* touchAllocation;
      if (_touchDataSize>0) {
	touchAllocation=new Block<Touch>(getBuffAllocationSize(_touchDataSize));
	touchAllocation->setCount(_touchDataSize);
	_tissueContext->_touchVector.push_back(touchAllocation);
	_recvbuf = touchAllocation->getData();
      }
      else _recvbuf = _touches;
    }
    else {
      delete [] _touches;
      _recvbuf = _touches = new Touch[getBuffAllocationSize(_touchDataSize)];
    }
    _previousTouchDataSize =  getUsableBuffSize(_touchDataSize);
  }
  else reinstateTouches();
}

int TouchAggregator::getNumberOfReceivePhasesPerCycle(int receiveCycle)
{
  assert(receiveCycle==0);
  return TOUCHAGGREGATOR_RECEIVE_PHASES;
}

void TouchAggregator::reinstateTouches() 
{
#ifndef LTWT_TOUCH
  Touch* touchEnd=_touches+_touchDataSize;
  for (Touch* touchPtr=_touches; touchPtr!=touchEnd; ++touchPtr) {
    touchPtr->reinstate();
  }
#endif
}

TouchAggregator::~TouchAggregator()
{
  delete [] _typeTouches; 
  delete [] _touchesPerSender;
  delete [] _touchDispls;
  delete [] _touches;
}
