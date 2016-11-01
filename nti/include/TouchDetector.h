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

#ifndef TOUCHDETECTOR_H
#define TOUCHDETECTOR_H

#define TOUCH_DETECTOR_RECEIVE_PHASES 2
#define TOUCH_DETECTOR_SEND_PHASES 2

#include <mpi.h>
#include "Touch.h"
#include "Sender.h"
#include "Receiver.h"
#include "TouchVector.h"
#include "TissueContext.h"
#include "ComputeBranch.h"

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <map>
#include <vector>
#include <list>

// 0..2 : TD_BEGIN_COORDS (3 doubles)
// 3..3 : TD_RADIUS (1 double)
// 4..4 : TD_KEY (1 doubles)
// 5..8 : TD_END_COORDS (3 doubles)
#define N_TD_DATA 8
#define TD_BEGIN_COORDS 0
#define TD_RADIUS 3
#define TD_KEY 4
#define TD_END_COORDS 5

class Decomposition;
class TouchSpace;
class TouchAnalyzer;
class NeuronPartitioner;
class Params;
class Capsule;
class ThreadUserData;
class Mutex;

class GlomeruliDetector;

class TouchDetector : public Sender, public Receiver
{
  public:
  TouchDetector(const int rank, const int nSlicers, const int nTouchDetectors,
                const int maxComputeOrder, const int nThreads,
                double appositionRate, Decomposition** decomposition,
                TouchSpace* detectionTouchSpace,
                TouchSpace* communicateTouchSpace,
                NeuronPartitioner* neuronPartitioner,
                TissueContext* tissueContext, Params* params);
  virtual ~TouchDetector();

  int getRank() { return _rank; }

  void prepareToReceive(int receiveCycle, int receivePhase,
                        CommunicatorFunction& funPtrRef);
  void* getRecvbuf(int receiveCycle, int receivePhase);
  int* getRecvcounts(int receiveCycle, int receivePhase);
  int* getRdispls(int receiveCycle, int receivePhase);
  MPI_Datatype* getRecvtypes(int receiveCycle, int receivePhase);
  int* getRecvSizes(int receiveCycle, int receivePhase);
  int getNumberOfReceiveCycles() { return 1; }
  int getNumberOfReceivePhasesPerCycle(int receiveCycle);
  void finalizeReceive(int receiveCycle, int receivePhase);
  int getNumberOfSenders() { return _numberOfSenders; }

  void prepareToSend(int sendCycle, int sendPhase,
                     CommunicatorFunction& funPtrRef);
  void* getSendbuf(int sendCycle, int sendPhase);
  int* getSendcounts(int sendCycle, int sendPhase);
  int* getSdispls(int sendCycle, int sendPhase);
  MPI_Datatype* getSendtypes(int sendCycle, int sendPhase);
  int getNumberOfSendCycles() { return 1; }
  int getNumberOfSendPhasesPerCycle(int sendCycle);
  void mergeWithSendBuf(int index, int count, int sendCycle, int sendPhase)
  {
    assert(0);
  }
  int getNumberOfReceivers() { return _numberOfReceivers; }

  void setTouchAnalyzer(TouchAnalyzer* touchAnalyzer);
  void resetTouchVector();
  TouchVector* getTouchVector() { return _touchVector; }
  void detectTouches();
  void doWork(int threadID, int i, ThreadUserData* data, Mutex* mutex);
  void doWork_original(int threadID, int i, ThreadUserData* data, Mutex* mutex);
  void doWork_new(int threadID, int i, ThreadUserData* data, Mutex* mutex);
  double findShortestDistance(Capsule& caps_from, Capsule& caps_to, Params*, int threadID,
		 double &sc, double&tc ) const;
  double findShortestDistance(double* coord, Capsule& caps_to, Params*, int threadID,
		 double&tc ) const;
  Capsule* getProximalCapsule(Capsule* caps, int sid); //caps = array, sid = index
  std::vector<Capsule*> getDistalCapsules(Capsule* caps, int sid);
  bool isNeighborsFormingTouchSIDside(int threadID, Params* params, double dist, Capsule* caps, int sid, int sid2);
  bool isNeighborsFormingTouchSID2side(int threadID, Params* params, double dist, Capsule* caps, int sid, int sid2);


  void writeToFile(std::string experimentName);
  void unique(bool unique);
  void resetBufferSize(bool resetBufferSize)
  {
    _resetBufferSize = resetBufferSize;
  }
  void receiveAtBufferOffset(bool receiveAtBufferOffset)
  {
    _receiveAtBufferOffset = receiveAtBufferOffset;
  }
  void setUpCapsules();
  void setCapsuleOffset(int offset);
  void setPass(TissueContext::DetectionPass detectionPass)
  {
    _detectionPass = detectionPass;
  }
  std::string getPassName();

  class TDSegment
  {
public:
    // TUAN: potential bug here if we change the keysize, RECOMMEND: move key to
    // the last component
    double seg[N_TD_DATA];
    double* getBeginCoords() { return &seg[TD_BEGIN_COORDS]; }
    double getRadius() { return seg[TD_RADIUS]; }
    key_size_t getKey() const { return seg[TD_KEY]; }
    double* getEndCoords() { return &seg[TD_END_COORDS]; }
    bool operator<(const TDSegment& s1) const;
    bool operator==(const TDSegment& s1) const;
  };

  private:
  void initializePhase1Receive(int);
  void initializePhase0Send();
  void initializePhase1Send();
  inline void distancePointLine(double p1A[3], double p1B[3], double p2A[3],
                                double p2B[3], double* pointLineDistances);

  int _numberOfSenders;
  int _numberOfReceivers;
  int _rank;
  int _nSlicers;
  int _nTouchDetectors;
  int _maxComputeOrder;
  int _nThreads;
  ThreadUserData* _threadUserData;

  double _appositionRate;
  Decomposition** _decomposition;
  TouchSpace* _detectionTouchSpace;
  TouchSpace* _communicateTouchSpace;
  NeuronPartitioner* _neuronPartitioner;
  Params* _params;
  bool _writeToFile;

  long long _connectionCount;
  TouchAnalyzer* _touchAnalyzer;

  // Receive Phase 0: ALLTOALL
  int* _segmentsPerSender;  // recvbuf
  int _numberOfInts;        // recvcount
  MPI_Datatype _typeInt;    // recvtype

  // Receive Phase 1: ALLTOALLW
  Capsule** _capsules;  // ptr to recvbuf array
  TouchVector* _touchVector; // each thread handle a TouchVector 
  TouchVector _initialTouchVector;

  int* _segmentCounts;          // recvcount
  int* _segmentDispls;          // recvDispls
  MPI_Datatype* _typeSegments;  // recvtype
  int _segmentDataSize;
  int _numberOfCapsules;
  int _previousPhase1RecvBufSize;

  // Send Phase 0: ALLTOALL
  int* _touchesPerReceiver;

  // Send Phase 1: ALLTOALLW
  Touch* _touchOrigin;
  int* _touchCounts;
  int* _touchDispls;
  MPI_Datatype* _typeTouches;
  double* _sendBuf;
  int _sendBufSize;
  int _maxNumTouches;
  int* _touchBlockLengths;
  MPI_Aint* _touchBlockDisplacements;

  bool _unique, _resetBufferSize, _receiveAtBufferOffset;
  TissueContext* _tissueContext;
  TissueContext::DetectionPass _detectionPass;
  std::string _experimentName;
};

#endif
