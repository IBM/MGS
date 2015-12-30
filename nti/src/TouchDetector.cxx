// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. and EPFL 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// ================================================================
//
// Geometric Tools, LLC
// Copyright (c) 1998-2011
// Distributed under the Boost Software License, Version 1.0.
// http://www.boost.org/LICENSE_1_0.txt
// http://www.geometrictools.com/License/Boost/LICENSE_1_0.txt
//
// Boost Software License - Version 1.0 - August 17th, 2003
//
// Permission is hereby granted, free of charge, to any person or organization
// obtaining a copy of the software and accompanying documentation covered by
// this license (the "Software") to use, reproduce, display, distribute,
// execute, and transmit the Software, and to prepare derivative works of the
// Software, and to permit third-parties to whom the Software is furnished to
// do so, all subject to the following:
//
// The copyright notices in the Software and this entire statement, including
// the above license grant, this restriction and the following disclaimer,
// must be included in all copies of the Software, in whole or in part, and
// all derivative works of the Software, unless such copies or derivative
// works are solely in the form of machine-executable object code generated by
// a source language processor.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
// SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
// FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.

#include "TouchDetector.h"
#include "TissueContext.h"
#include "Segment.h"
#include "Decomposition.h"
#include "TouchAnalyzer.h"
#include "TouchSpace.h"
#include "SegmentDescriptor.h"
#include "NeuronPartitioner.h"
#include "Tissue.h"
#include "TouchAggregator.h"
#include "VecPrim.h"
#include "Params.h"
#include "Capsule.h"
#include "RNG.h"
#include "rndm.h"
#include "BuffFactor.h"
#include "Utilities.h"
#ifndef DISABLE_PTHREADS
#include "For.h"
#endif
#include "ThreadUserData.h"
#ifdef USING_BLUEGENE
#include "BG_AvailableMemory.h"
#endif

#include <cassert>
#include <algorithm>
#include <memory>

#define SMALL_NUM 0.00000000000001  // anything that avoids division overflow

TouchDetector::TouchDetector(
    const int rank, const int nSlicers, const int nTouchDetectors,
    const int maxComputeOrder, const int nThreads, double appositionRate,
    Decomposition** decomposition, TouchSpace* detectionTouchSpace,
    TouchSpace* communicateTouchSpace, NeuronPartitioner* neuronPartitioner,
    TissueContext* tissueContext, Params* params)
    : _numberOfSenders(0),
      _numberOfReceivers(0),
      _rank(rank),
      _nSlicers(nSlicers),
      _nTouchDetectors(nTouchDetectors),
      _maxComputeOrder(maxComputeOrder),
      _nThreads(nThreads),
      _threadUserData(0),
      _appositionRate(appositionRate),
      _decomposition(decomposition),
      _detectionTouchSpace(detectionTouchSpace),
      _communicateTouchSpace(communicateTouchSpace),
      _neuronPartitioner(neuronPartitioner),
      _params(params),
      _writeToFile(false),
      _connectionCount(0),
      _touchAnalyzer(0),
      _numberOfInts(1),
      _typeInt(MPI_INT),
      _capsules(0),
      _touchVector(0),
      _segmentsPerSender(0),
      _segmentDispls(0),
      _typeSegments(0),
      _segmentDataSize(0),
      _numberOfCapsules(0),
      _previousPhase1RecvBufSize(1),
      _touchesPerReceiver(0),
      _touchOrigin(0),
      _touchCounts(0),
      _touchDispls(0),
      _typeTouches(0),
      _sendBuf(0),
      _sendBufSize(1),
      _maxNumTouches(1),
      _touchBlockLengths(0),
      _touchBlockDisplacements(0),
      _unique(true),
      _resetBufferSize(true),
      _receiveAtBufferOffset(false),
      _tissueContext(tissueContext),
      _detectionPass(TissueContext::FIRST_PASS),
      _experimentName("") {
  _numberOfSenders = _numberOfReceivers =
      (nSlicers > nTouchDetectors) ? nSlicers : nTouchDetectors;

  Capsule capsule;
#ifdef A2AW
  MPI_Aint capsuleAddress, disp;
  MPI_Get_address(&capsule, &capsuleAddress);
  MPI_Get_address(capsule.getData(), &disp);
  disp -= capsuleAddress;
  MPI_Datatype typeCapsuleBasic;
  int blocklen = N_CAP_DATA;
  MPI_Type_create_hindexed(1, &blocklen, &disp, MPI_DOUBLE, &typeCapsuleBasic);
  _typeSegments = new MPI_Datatype[_numberOfSenders];
  MPI_Datatype typeCapsule;
  MPI_Type_create_resized(typeCapsuleBasic, 0, sizeof(Capsule), &typeCapsule);
  MPI_Type_commit(&typeCapsule);
#else
  _typeSegments = new MPI_Datatype[1];
  Datatype datatype(3, &capsule);
  datatype.set(0, MPI_LB, 0);
  datatype.set(1, MPI_DOUBLE, N_CAP_DATA, capsule.getData());
  datatype.set(2, MPI_UB, sizeof(Capsule));
  _typeSegments[0] = datatype.commit();
  _sendBuf = new double[_sendBufSize];
#endif

  _segmentsPerSender = new int[_numberOfSenders];
  _segmentDispls = new int[_numberOfSenders];

  for (int i = 0; i < _numberOfSenders; ++i) {
#ifdef A2AW
    _typeSegments[i] = typeCapsule;
#endif
    _segmentsPerSender[i] = 0;
    _segmentDispls[i] = 0;
  }

  if (_tissueContext) {
    _capsules = &_tissueContext->_capsules;
    assert(_communicateTouchSpace == 0);
  } else
    _capsules = new Capsule*;

  *_capsules = new Capsule[_previousPhase1RecvBufSize];

  if (_communicateTouchSpace || _tissueContext) {
    _touchVector = &_initialTouchVector;
    _touchesPerReceiver = new int[_numberOfReceivers];
    _touchCounts = new int[_numberOfReceivers];
    _touchDispls = new int[_numberOfReceivers];
#ifdef A2AW
    _typeTouches = new MPI_Datatype[_numberOfReceivers];
#else
    _typeTouches = new MPI_Datatype[1];
#endif
    for (int i = 0; i < _numberOfReceivers; ++i) {
      _touchesPerReceiver[i] = 0;
      _touchCounts[i] = 1;
      _touchDispls[i] = 0;
    }

#ifdef A2AW
    _touchBlockLengths = new int[_maxNumTouches];
    _touchBlockDisplacements = new MPI_Aint[_maxNumTouches];

    for (int j = 0; j < _maxNumTouches; ++j) {
      _touchBlockLengths[j] = 1;
      _touchBlockDisplacements[j] = 0;
    }

    for (int i = 0; i < _numberOfReceivers; ++i) {
      int numTouches = 1;
      MPI_Type_create_hindexed(numTouches, _touchBlockLengths,
                               _touchBlockDisplacements,
                               *(Touch::getTypeTouch()), &_typeTouches[i]);
      MPI_Type_commit(&_typeTouches[i]);
    }
#else
    MPI_Type_contiguous(N_TOUCH_DATA, MPI_DOUBLE, _typeTouches);
    MPI_Type_commit(_typeTouches);
#endif
  }
  _threadUserData = new ThreadUserData(_nThreads);
  for (int i = 0; i < _nThreads; ++i) {
    _threadUserData->_parms[i] = new Params(*_params);
    _threadUserData->_touchSpaces[i] = _detectionTouchSpace->duplicate();
  }
}

TouchDetector::~TouchDetector() {
  delete[] _typeSegments;
  delete[] _segmentsPerSender;
  delete[] _segmentDispls;
  delete[] _touchesPerReceiver;
  delete[] _touchCounts;
  delete[] _touchDispls;
  delete[] _typeTouches;
  delete[] _touchBlockLengths;
  delete[] _touchBlockDisplacements;
  if (_tissueContext == 0) {
    delete[] * _capsules;
    delete _capsules;
  }
  delete _threadUserData;
}

void TouchDetector::prepareToReceive(int receiveCycle, int receivePhase,
                                     CommunicatorFunction& funPtrRef) {
  switch (receivePhase) {
    case 0:
      funPtrRef = &Communicator::allToAll;
      break;
    case 1:
      initializePhase1Receive(receiveCycle);
#ifdef A2AW
      funPtrRef = &Communicator::allToAllW;
#else
      funPtrRef = &Communicator::allToAllV;
#endif
      break;
    default:
      assert(0);
  }
}

void* TouchDetector::getRecvbuf(int receiveCycle, int receivePhase) {
  void* rval;
  switch (receivePhase) {
    case 0:
      rval = (void*)_segmentsPerSender;
      break;
    case 1:
      if (_receiveAtBufferOffset)
        rval = (void*)((*_capsules) + _numberOfCapsules);
      else
        rval = (void*)(*_capsules);
      break;
    default:
      assert(0);
  }
  return rval;
}

int* TouchDetector::getRecvcounts(int receiveCycle, int receivePhase) {
  int* rval;
  switch (receivePhase) {
    case 0:
      rval = &_numberOfInts;
      break;
    case 1:
      rval = _segmentsPerSender;
      break;
    default:
      assert(0);
  }
  return rval;
}

int* TouchDetector::getRdispls(int receiveCycle, int receivePhase) {
  int* rval;
  switch (receivePhase) {
    case 0:
      assert(0);
      break;
    case 1:
      rval = _segmentDispls;
      break;
    default:
      assert(0);
  }
  return rval;
}

MPI_Datatype* TouchDetector::getRecvtypes(int receiveCycle, int receivePhase) {
  MPI_Datatype* rval;
  switch (receivePhase) {
    case 0:
      rval = &_typeInt;
      break;
    case 1:
      rval = _typeSegments;
      break;
    default:
      assert(0);
  }
  return rval;
}

void TouchDetector::initializePhase1Receive(int receiveCycle) {
  if (_resetBufferSize) _segmentDataSize = 0;
  int segCount = 0;
  for (int i = 0; i < _numberOfSenders; i++) {
#ifdef A2AW
    _segmentDispls[i] = segCount * sizeof(Capsule);
#else
    _segmentDispls[i] = segCount;
#endif
    segCount += _segmentsPerSender[i];
  }
  _segmentDataSize += segCount;
  if (_segmentDataSize > _previousPhase1RecvBufSize) {
    delete[] * _capsules;
    *_capsules = new Capsule[getBuffAllocationSize(_segmentDataSize)];
    _previousPhase1RecvBufSize = getUsableBuffSize(_segmentDataSize);
  }
}

void TouchDetector::prepareToSend(int sendCycle, int sendPhase,
                                  CommunicatorFunction& funPtrRef) {
  assert(sendCycle == 0);
  switch (sendPhase) {
    case 0:
      initializePhase0Send();
      funPtrRef = &Communicator::allToAll;
      break;
    case 1:
      initializePhase1Send();
#ifdef A2AW
      funPtrRef = &Communicator::allToAllW;
#else
      funPtrRef = &Communicator::allToAllV;
#endif
      break;
    default:
      assert(0);
  }
}

void* TouchDetector::getSendbuf(int sendCycle, int sendPhase) {
  assert(sendCycle == 0);
  void* rval;
  switch (sendPhase) {
    case 0:
      rval = (void*)_touchesPerReceiver;
      break;
#ifdef A2AW
    case 1:
      rval = (void*)_touchOrigin;
      break;
#else
    case 1:
      rval = (void*)_sendBuf;
      break;
#endif
    default:
      assert(0);
  }
  return rval;
}

int* TouchDetector::getSendcounts(int sendCycle, int sendPhase) {
  assert(sendCycle == 0);
  int* rval;
  switch (sendPhase) {
    case 0:
      rval = &_numberOfInts;
      break;
#ifdef A2AW
    case 1:
      rval = _touchCounts;
      break;
#else
    case 1:
      rval = _touchesPerReceiver;
      break;
#endif
    default:
      assert(0);
  }
  return rval;
}

int* TouchDetector::getSdispls(int sendCycle, int sendPhase) {
  assert(sendCycle == 0);
  int* rval;
  switch (sendPhase) {
    case 0:
      assert(0);
      break;
    case 1:
      rval = _touchDispls;
      break;
    default:
      assert(0);
  }
  return rval;
}

MPI_Datatype* TouchDetector::getSendtypes(int sendCycle, int sendPhase) {
  assert(sendCycle == 0);
  MPI_Datatype* rval;
  switch (sendPhase) {
    case 0:
      rval = &_typeInt;
      break;
    case 1:
      rval = _typeTouches;
      break;
    default:
      assert(0);
  }
  return rval;
}

int TouchDetector::getNumberOfSendPhasesPerCycle(int sendCycle) {
  assert(sendCycle == 0);
  return TOUCH_DETECTOR_SEND_PHASES;
}

void TouchDetector::initializePhase0Send() {
  assert(_touchVector);
  std::map<int, std::list<TouchIndex> >& touchMap = _touchVector->getTouchMap();
  int size = 0;
  std::map<int, std::list<TouchIndex> >::iterator mapIter,
      mapEnd = touchMap.end();
  for (int i = 0; i < _numberOfReceivers; ++i) {
    mapIter = touchMap.find(i);
    if (mapIter == mapEnd)
      _touchesPerReceiver[i] = 0;
    else
      _touchesPerReceiver[i] = mapIter->second.size();
#ifdef A2AW
    if (_touchesPerReceiver[i] > size) size = _touchesPerReceiver[i];
#else
    _touchDispls[i] = size;
    size += _touchesPerReceiver[i];
#endif
  }

#ifdef A2AW
  if (_maxNumTouches < size) {
    int touchAllocationSize = getBuffAllocationSize(size);
    delete[] _touchBlockLengths;
    delete[] _touchBlockDisplacements;
    _touchBlockLengths = new int[touchAllocationSize];
    _touchBlockDisplacements = new MPI_Aint[touchAllocationSize];
    for (int j = 0; j < touchAllocationSize; ++j) {
      _touchBlockLengths[j] = 0;
      _touchBlockDisplacements[j] = 0;
    }
    _maxNumTouches = getUsableBuffSize(size);
  }
#else
  size *= N_TOUCH_DATA;
  if (size > _sendBufSize) {
    delete[] _sendBuf;
    _sendBuf = new double[getBuffAllocationSize(size)];
    _sendBufSize = getUsableBuffSize(size);
  }
#endif
}

void TouchDetector::initializePhase1Send() {
  std::map<int, std::list<TouchIndex> >& touchMap = _touchVector->getTouchMap();
  std::map<int, std::list<TouchIndex> >::iterator mapIter,
      mapEnd = touchMap.end();
#ifdef A2AW
  MPI_Aint touchOriginAddress;
  _touchOrigin = _touchVector->getTouchOrigin();
  MPI_Get_address(_touchOrigin, &touchOriginAddress);
  MPI_Aint touchAddress;
  int numBlocks = 0;
  for (int i = 0; i < _numberOfReceivers; ++i) {
    mapIter = touchMap.find(i);
    int j = 0;
    if (mapIter != mapEnd) {
      std::list<TouchIndex>::iterator iter = mapIter->second.begin(),
                                      end = mapIter->second.end();
      while (iter != end) {
        Touch& t = _touchVector->getValue(*iter);
        int number = iter->getBlock(), index = iter->getIndex();
        MPI_Aint touchAddress;
        MPI_Get_address(&t, &touchAddress);
        _touchBlockLengths[j] = 1;
        _touchBlockDisplacements[j] = touchAddress - touchOriginAddress;
        while (++iter != end && iter->getBlock() == number &&
               iter->getIndex() == ++index)
          ++_touchBlockLengths[j];
        ++j;
      }
    }
    MPI_Type_free(&_typeTouches[i]);
    MPI_Type_create_hindexed(j, _touchBlockLengths, _touchBlockDisplacements,
                             *(Touch::getTypeTouch()), &_typeTouches[i]);
    MPI_Type_commit(&_typeTouches[i]);
  }
#else
  int writePos = 0;
  for (int i = 0; i < _numberOfReceivers; ++i) {
    mapIter = touchMap.find(i);
    if (mapIter != mapEnd) {
      std::list<TouchIndex>::iterator iter = mapIter->second.begin(),
                                      end = mapIter->second.end();
      for (; iter != end; ++iter) {
        assert(writePos <= _sendBufSize - N_TOUCH_DATA);
        Touch& t = _touchVector->getValue(*iter);
        std::copy(t.getTouchData(), t.getTouchData() + N_TOUCH_DATA,
                  &_sendBuf[writePos]);
        // memcpy(&_sendBuf[writePos], t.getTouchData(),
        // N_TOUCH_DATA*sizeof(double));
        writePos += N_TOUCH_DATA;
      }
    }
  }
#endif
}

void TouchDetector::setTouchAnalyzer(TouchAnalyzer* touchAnalyzer) {
  _touchAnalyzer = touchAnalyzer;
}

void TouchDetector::resetTouchVector() {
  // printf("%d,
  // %d\n",_initialTouchVector.getCount(),_tissueContext->_touchVector.getCount());
  _initialTouchVector.clear();
  if (_tissueContext) _touchVector = &_tissueContext->_touchVector;
}

void TouchDetector::setUpCapsules() {
  if (_tissueContext) {
    _numberOfCapsules = _segmentDataSize = _tissueContext->setUpCapsules(
        _segmentDataSize, _detectionPass, _rank, _maxComputeOrder);
  } else {
    Capsule* caps = *_capsules, *capsEnd = caps + _segmentDataSize;
    std::sort(caps, capsEnd);
    capsEnd = std::unique(caps, capsEnd);
    _segmentDataSize = _numberOfCapsules = capsEnd - caps;
  }
}

void TouchDetector::setCapsuleOffset(int offset) {
  *_capsules += offset;
  _segmentDataSize -= offset;
  _numberOfCapsules -= offset;
  if (_tissueContext) _tissueContext->_nCapsules -= offset;
}

void TouchDetector::detectTouches() {
#ifdef USING_BLUEGENE
  if (_rank == 0)
    printf("Available memory before touch detection %s : %lf MB.\n\n",
           getPassName().c_str(), AvailableMemory());
#endif
  if (_unique) assert((*_decomposition)->isCoordinatesBased());
  _touchVector->clear();
  RNG* rng;
  if (_tissueContext)
    rng = &_tissueContext->_touchSampler;
  else {
    rng = new RNG;
    rng->reSeed(12345678, _rank);
  }
  for (int i = 0; i < _nThreads; ++i) {
    _threadUserData->_touchVectors[i].clear();
    _threadUserData->_rangens[i].reSeed(lrandom(*rng), _rank);
  }
  if (!_tissueContext) delete rng;

  long int binpos = 0;
  _connectionCount = 0;
  FILE* data = 0;
  if (_writeToFile) {
    char filename[256];
    sprintf(filename, "outTouches_%s_%d.bin", _experimentName.c_str(), _rank);
    if ((data = fopen(filename, "wb")) == NULL) {
      printf("Could not open the output file %s!\n", filename);
      MPI_Finalize();
      exit(0);
    }
    _threadUserData->_file = data;
    int t = 1;
    fwrite(&t, sizeof(int), 1, data);
    binpos = ftell(data);
    fwrite(&_connectionCount, sizeof(long long), 1, data);
  }

  _threadUserData->resetDecompositions(*_decomposition);

#ifndef DISABLE_PTHREADS
  For<TouchDetector, ThreadUserData>::execute(0, 1, _numberOfCapsules, this,
                                              _threadUserData, _nThreads);
#else
  for (int sid = 0; sid < _numberOfCapsules; ++sid) {
    doWork(0, sid, _threadUserData, 0);
  }  // for (; sid<_nCapsules...
#endif
  _connectionCount = 0;
  for (int i = 0; i < _nThreads; ++i) {
    _connectionCount += _threadUserData->_touchVectors[i].getCount();
    _touchVector->merge(_threadUserData->_touchVectors[i]);
  }
  assert(_connectionCount == _touchVector->getCount());
  if (_tissueContext == 0) {
    Capsule* caps = *_capsules;
    std::map<double, int> capsuleMap;
    for (int sid = 0; sid < _numberOfCapsules; ++sid)
      capsuleMap[caps[sid].getKey()] = sid;
    TouchVector::TouchIterator titer = _touchVector->begin(),
                               tend = _touchVector->end();
    for (; titer != tend; ++titer) {
      if (_communicateTouchSpace) {
        double s1Key = titer->getKey1();
        if (_communicateTouchSpace->isInSpace(s1Key))
          _touchVector->mapTouch(
              _neuronPartitioner->getRank(caps[capsuleMap[s1Key]].getSphere()),
              titer);
        double s2Key = titer->getKey2();
        if (_communicateTouchSpace->isInSpace(s2Key))
          _touchVector->mapTouch(
              _neuronPartitioner->getRank(caps[capsuleMap[s2Key]].getSphere()),
              titer);
      }
      if (_touchAnalyzer != 0) {
        _touchAnalyzer->evaluateTouch(*titer);
      }
    }
  }

  if (_touchAnalyzer) _touchAnalyzer->confirmTouchCounts(_connectionCount);

  if (_writeToFile) {
    fseek(data, binpos, SEEK_SET);
    fwrite(&_connectionCount, sizeof(long long), 1, data);
    fclose(data);
  }
  long long totalCount;
  MPI_Allreduce((void*)&_connectionCount, (void*)&totalCount, 1, MPI_LONG_LONG,
                MPI_SUM, MPI_COMM_WORLD);
  if (_rank == 0) printf("Total Touch = %lld\n\n", totalCount);
#ifdef USING_BLUEGENE
  if (_rank == 0) {
    printf("Available memory after touch detection %s : %lf MB.\n\n",
           getPassName().c_str(), AvailableMemory());
  }
#endif
}

// GOAL: Find the Capsules that 'touches' the given Capsule
//    sid = the index of the given Capsule
//
void TouchDetector::doWork(int threadID, int sid, ThreadUserData* data,
                           Mutex* mutex) {
  Params* params = data->_parms[threadID];
  TouchVector& touchVector = _threadUserData->_touchVectors[threadID];
  RNG& rng = _threadUserData->_rangens[threadID];
  TouchSpace* detectionTouchSpace = _threadUserData->_touchSpaces[threadID];
  Decomposition* decomposition = _threadUserData->_decompositions[threadID];
  Capsule* caps = *_capsules;
  double s1Ax = 0, s1Ay = 0, s1Az = 0, s1Bx = 0, s1By = 0, s1Bz = 0, s1Ar = 0,
         s1Ab = 0, s2Ax = 0, s2Ay = 0, s2Az = 0, s2Bx = 0, s2By = 0, s2Bz = 0,
         s2Ar = 0, s2Ab = 0, a = 0, b = 0, c = 0, d = 0, e = 0, D = 0, sc = 0,
         sN = 0, sD = 0, tc = 0, tN = 0, tD = 0, u0 = 0, u1 = 0, u2 = 0, v0 = 0,
         v1 = 0, v2 = 0, w0 = 0, w1 = 0, w2 = 0, aw0 = 0, aw1 = 0, aw2 = 0,
         bw0 = 0, bw1 = 0, bw2 = 0, ww0 = 0, ww1 = 0, ww2 = 0, disSphere = 0,
         mdis = 0, mdis2 = 0, dist, crit, c1, c2;
  double pointLineDistances[4];
  // loop through all other Capsules
  for (int sid2 = 0; sid2 < _numberOfCapsules; ++sid2)
	  // check prob. for forming a touch
    if (_appositionRate >= 1.0 || drandom(rng) < _appositionRate) {
      key_size_t s2Key, s1Key = caps[sid].getKey();
      double* s1begin = caps[sid].getBeginCoordinates();
      double* s1end = caps[sid].getEndCoordinates();
      s1Ar = caps[sid].getRadius() + params->getRadius(s1Key);

      s1Ax = s1begin[0];
      s1Ay = s1begin[1];
      s1Az = s1begin[2];
      s1Bx = s1end[0];
      s1By = s1end[1];
      s1Bz = s1end[2];

      u0 = s1Bx - s1Ax;  // FOR USE BELOW : OPTIMIZATION
      u1 = s1By - s1Ay;  // FOR USE BELOW : OPTIMIZATION
      u2 = s1Bz - s1Az;  // FOR USE BELOW : OPTIMIZATION
      s1Ab = sqrt(SqDist(s1begin, s1end)) +
             2.0 * s1Ar;                // FOR USE BELOW : OPTIMIZATION
      a = u0 * u0 + u1 * u1 + u2 * u2;  // FOR USE BELOW : OPTIMIZATION

      s2Key = caps[sid2].getKey();
	  //check if both capsules can be grouped to form 'electricalSynapse'
	  //                            OR                'chemicalSynapse'
      if (detectionTouchSpace->areInSpace(s1Key, s2Key)) {
        double* s2begin = caps[sid2].getBeginCoordinates();
        double* s2end = caps[sid2].getEndCoordinates();
        s2Ar = caps[sid2].getRadius() + params->getRadius(s2Key);
        s2Ax = s2begin[0];
        s2Ay = s2begin[1];
        s2Az = s2begin[2];
        s2Bx = s2end[0];
        s2By = s2end[1];
        s2Bz = s2end[2];

        // Check spheres if they are close first

        // s1Ab = SqDist(s1, &s1[TD_END_COORDS])+2.0*s1Ar; COMPUTED ABOVE :
        // OPTIMIZATION
        s2Ab = sqrt(SqDist(s2begin, s2end)) + 2.0 * s2Ar;
        disSphere = (s1Ab + s2Ab) * (s1Ab + s2Ab);

        w0 = s1Ax - s2Ax;
        w1 = s1Ay - s2Ay;
        w2 = s1Az - s2Az;
        ww0 = s1Bx - s2Bx + w0;
        ww1 = s1By - s2By + w1;
        ww2 = s1Bz - s2Bz + w2;

        if (((ww0 * ww0 + ww1 * ww1 + ww2 * ww2) <=
             disSphere)) {  // it is 0.5*0.5

          // u0 = s1Bx - s1Ax;   COMPUTED ABOVE : OPTIMIZATION
          // u1 = s1By - s1Ay;   COMPUTED ABOVE : OPTIMIZATION
          // u2 = s1Bz - s1Az;   COMPUTED ABOVE : OPTIMIZATION
          v0 = s2Bx - s2Ax;
          v1 = s2By - s2Ay;
          v2 = s2Bz - s2Az;

          // double a = dot(u,u); //always >= 0
          // a = u0 * u0 + u1 * u1 + u2 * u2      COMPUTED ABOVE : OPTIMIZATION
          // double b = dot(u,v);
          b = u0 * v0 + u1 * v1 + u2 * v2;
          // double c = dot(v,v); //always >= 0
          c = v0 * v0 + v1 * v1 + v2 * v2;
          // double d = dot(u,w);
          d = u0 * w0 + u1 * w1 + u2 * w2;
          // double e = dot(v,w);
          e = v0 * w0 + v1 * w1 + v2 * w2;
          D = a * c - b * b;  // always >= 0
          sD = D;             // sc = sN / sD, default sD = D >= 0
          tD = D;             // tc = tN / tD, default tD = D >= 0

          // compute the line parameters of the two closest points
          if (D < SMALL_NUM) {  // the lines are almost parallel
            sN = 0.0;  // force using point P0 on segment S1
            sD = 1.0;  // to prevent possible division by 0.0 later
            tN = e;
            tD = c;
          } else {  // get the closest points on the infinite lines
            sN = (b * e - c * d);
            tN = (a * e - b * d);
            if (sN <= 0.0) {  // sc < 0 => the s=0 edge is visible
              sN = 0.0;
              tN = e;
              tD = c;
            } else if (sN >= sD) {  // sc > 1 => the s=1 edge is visible
              sN = sD;
              tN = e + b;
              tD = c;
            }
          }

          if (tN <= 0.0) {  // tc < 0 => the t=0 edge is visible
            tN = 0.0;
            // recompute sc for this edge
            if (-d <= 0.0)
              sN = 0.0;
            else if (-d >= a)
              sN = sD;
            else {
              sN = -d;
              sD = a;
            }
          } else if (tN >= tD) {  // tc > 1 => the t=1 edge is visible
            tN = tD;  // recompute sc for this edge
            if ((-d + b) <= 0.0)
              sN = 0.0;
            else if ((-d + b) >= a)
              sN = sD;
            else {
              sN = (-d + b);
              sD = a;
            }
          }
          if (fabs(sN) < SMALL_NUM)
            aw0 = aw1 = aw2 = sc = 0.0;
          else {
            sc = sN / sD;
            aw0 = u0 * sc;
            aw1 = u1 * sc;
            aw2 = u2 * sc;
            w0 += aw0;
            w1 += aw1;
            w2 += aw2;
          }
          if (fabs(tN) < SMALL_NUM)
            bw0 = bw1 = bw2 = tc = 0.0;
          else {
            tc = tN / tD;
            bw0 = v0 * tc;
            bw1 = v1 * tc;
            bw2 = v2 * tc;
            w0 -= bw0;
            w1 -= bw1;
            w2 -= bw2;
          }
          dist = w0 * w0 + w1 * w1 + w2 * w2;
          crit = (s1Ar + s2Ar) * (s1Ar + s2Ar);
          bool countTouch = false;
          if (dist < crit) {
            /**
             * The coordinates of the touch can be calculated by vector
             * arithmetic as follows:
             * touchX = s1Ax + aw0 - w0*0.5;
             * touchY = s1Ay + aw1 - w1*0.5;
             * touchZ = s1Az + aw2 - w2*0.5;
             */

            countTouch = true;

            if (_unique) {
// The following code ensures global non-redundant touches by setting
// the boolean "countTouch"

#if 1
              ShallowArray<int, MAXRETURNRANKS, 100> ranks1, ranks2;
              decomposition->getRanks(&caps[sid].getSphere(),
                                      caps[sid].getEndCoordinates(),
                                      params->getRadius(s1Key), ranks1);
              decomposition->getRanks(&caps[sid2].getSphere(),
                                      caps[sid2].getEndCoordinates(),
                                      params->getRadius(s2Key), ranks2);
              ranks1.merge(ranks2);

              countTouch = false;
              ShallowArray<int, MAXRETURNRANKS, 100>::iterator
                  ranksIter = ranks1.begin(),
                  ranksEnd = ranks1.end();
              if (ranksIter != ranksEnd) {
                int idx = *ranksIter;
                ++ranksIter;
                for (; ranksIter != ranksEnd; ++ranksIter) {
                  if (idx == *ranksIter) {
                    if (idx == _rank) countTouch = true;
                    break;
                  }
                  idx = *ranksIter;
                }
              }
#else
              Sphere sphere1;
              sphere1._coords[0] = s1Ax + aw0;
              sphere1._coords[1] = s1Ay + aw1;
              sphere1._coords[2] = s1Az + aw2;
              int touchVolume = decomposition->getRank(sphere1);
              countTouch = (touchVolume == _rank);
              if (!countTouch) {
                Sphere sphere2;
                sphere2._coords[0] = s2Ax + bw0;
                sphere2._coords[1] = s2Ay + bw1;
                sphere2._coords[2] = s2Az + bw2;
                countTouch = (decomposition->getRank(sphere2) == _rank &&
                              !decomposition->mapsToRank(
                                   &caps[sid2].getSphere(), s2end,
                                   params->getRadius(s2Key), touchVolume));
              }
#endif
            }
            if (countTouch) {
              Touch touch;
              touch.setKey1(s1Key);
              touch.setKey2(s2Key);

              touch.setProp1(sc);  // aw0*aw0+aw1*aw1+aw2*aw2);
              touch.setProp2(tc);  // bw0*bw0+bw1*bw1+bw2*bw2);
#ifndef LTWT_TOUCH
              touch.setDistance(dist);
              distancePointLine(s1begin, s1end, s2begin, s2end,
                                pointLineDistances);
              short* endTouch = touch.getEndTouches();
			  //with 2 ends for each Capsule, 
			  //... we have 4 pairs of points to find the distance
              for (int i = 0; i < 4; ++i) {
                endTouch[i] = 0;
                if (pointLineDistances[i] < crit) endTouch[i] = 1;
              }
#endif
              touchVector.push_back(touch, mutex);
              if (_writeToFile) {
#ifndef DISABLE_PTHREADS
                mutex->lock();
#endif
                touch.writeToFile(data->_file);
#ifndef DISABLE_PTHREADS
                mutex->unlock();
#endif
              }
            }  // if (countTouch)
          }    // if (dist<crit)
        }  // if((ww0 * ww0 + ww1 * ww1 + ww2
      }  // if (detectionTouchSpace->areInSpace...
    }  // for (; sid2<_nCapsules...
}

void TouchDetector::distancePointLine(double p1A[3], double p1B[3],
                                      double p2A[3], double p2B[3],
                                      double* pointLineDistances) {
  double dis, U, Intersection[3], v[3];
  double* p[3];

  for (int i = 0; i < 4; ++i) {
    switch (i) {
      case 0:
        p[0] = p1A;
        p[1] = p2A;
        p[2] = p2B;
        break;
      case 1:
        p[0] = p1B;
        p[1] = p2A;
        p[2] = p2B;
        break;
      case 2:
        p[0] = p2A;
        p[1] = p1A;
        p[2] = p1B;
        break;
      case 3:
        p[0] = p2B;
        p[1] = p1A;
        p[2] = p1B;
        break;
      default:
        break;
    }

    // find distance
    v[0] = p[1][0] - p[2][0];
    v[1] = p[1][1] - p[2][1];
    v[2] = p[1][2] - p[2][2];
    // dis = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    dis = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];

    U = (((p[0][0] - p[1][0]) * (-v[0])) + ((p[0][1] - p[1][1]) * (-v[1])) +
         ((p[0][2] - p[1][2]) * (-v[2]))) /
        //( dis * dis );
        (dis);

    if (U < 0.0f) {
      // find distance
      v[0] = p[0][0] - p[1][0];
      v[1] = p[0][1] - p[1][1];
      v[2] = p[0][2] - p[1][2];
      pointLineDistances[i] = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
      continue;
    }
    if (U > 1.0f) {
      // find distance
      v[0] = p[0][0] - p[2][0];
      v[1] = p[0][1] - p[2][1];
      v[2] = p[0][2] - p[2][2];
      pointLineDistances[i] = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
      continue;
    }

    Intersection[0] = p[1][0] - U * (v[0]);
    Intersection[1] = p[1][1] - U * (v[1]);
    Intersection[2] = p[1][2] - U * (v[2]);

    // find distance
    v[0] = Intersection[0] - p[0][0];
    v[1] = Intersection[1] - p[0][1];
    v[2] = Intersection[2] - p[0][2];
    pointLineDistances[i] = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    // pointLineDistance[i] = Magnitude( Point, &Intersection );
  }
}

int TouchDetector::getNumberOfReceivePhasesPerCycle(int receiveCycle) {
  return TOUCH_DETECTOR_RECEIVE_PHASES;
}

void TouchDetector::finalizeReceive(int receiveCycle, int receivePhase) {
  assert(receiveCycle == 0);
  switch (receivePhase) {
    case 1:
      if (_tissueContext == 0) {
        setUpCapsules();
        detectTouches();
      }
  }
}

void TouchDetector::writeToFile(std::string experimentName) {
  _writeToFile = true;
  _experimentName = experimentName;
}

void TouchDetector::unique(bool unique) { _unique = unique; }

std::string TouchDetector::getPassName() {
  std::string rval;
  if (_detectionPass == TissueContext::FIRST_PASS)
    rval = "first pass";
  else if (_detectionPass == TissueContext::SECOND_PASS)
    rval = "second pass";
  else
    rval = "additional pass";
  return rval;
}
