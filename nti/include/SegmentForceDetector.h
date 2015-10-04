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

#ifndef SEGMENTFORCEDETECTOR_H
#define SEGMENTFORCEDETECTOR_H

#define SEGMENTFORCE_DETECTOR_RECEIVE_PHASES 2
#define SEGMENTFORCE_DETECTOR_SEND_PHASES 3
#define SEGMENTFORCE_ALLOCATION 100

#include <mpi.h>
#include "SegmentDescriptor.h"
#include "SegmentForce.h"
#include "Params.h"

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <list>

#include "Receiver.h"
#include "Sender.h"

class Decomposition;
class TouchSpace;
class NeuronPartitioner;
class Capsule;
class Mutex;
class ThreadUserData;

class SegmentForceDetector : public Receiver, public Sender
{
   public:

     SegmentForceDetector(const int rank, const int nSlicers, const int nSegmentForceDetectors, int nThreads, 
		       Decomposition** decomposition, TouchSpace* detectionSegmentForceSpace, 
		       NeuronPartitioner* neuronPartitioner, Params* params);
     virtual ~SegmentForceDetector();
             
     int getRank() {return _rank;}

     void prepareToReceive(int receiveCycle, int receivePhase, CommunicatorFunction& funPtrRef);
     void* getRecvbuf(int receiveCycle, int receivePhase);
     int* getRecvcounts(int receiveCycle, int receivePhase);
     int* getRdispls(int receiveCycle, int receivePhase);
     MPI_Datatype* getRecvtypes(int receiveCycle, int receivePhase);
     int* getRecvSizes(int receiveCycle, int receivePhase);
     int getNumberOfReceiveCycles() {return 1;}
     int getNumberOfReceivePhasesPerCycle(int receiveCycle);
     void finalizeReceive(int receiveCycle, int receivePhase) {}
     int getNumberOfSenders() {return _numberOfSenders;}

     void prepareToSend(int sendCycle, int sendPhase, CommunicatorFunction& funPtrRef);
     void* getSendbuf(int sendCycle, int sendPhase);
     int* getSendcounts(int sendCycle, int sendPhase);
     int* getSdispls(int sendCycle, int sendPhase);
     MPI_Datatype* getSendtypes(int sendCycle, int sendPhase);
     int getNumberOfSendCycles() {return 1;}
     int getNumberOfSendPhasesPerCycle(int sendCycle);
     void mergeWithSendBuf(int index, int count, int sendCycle, int sendPhase) {assert(0);}
     int getNumberOfReceivers() {return _numberOfReceivers;}

     void detectSegmentForces();
     void computeForces(Capsule* s1, Capsule* s2, Params* parms, double* Fa, double& E0, double EpsA, double sigmaA);
     void doWork(int threadID, int i, ThreadUserData* data, Mutex* mutex);
     void writeToFile() {_writeToFile=true;}
     double& getEnergy() {return _E0;}
     void updateCoveredSegments(bool updateCoveredSegments) {
       _updateCoveredSegments = updateCoveredSegments;
     }

   private:

     void initializePhase1Receive();

     int _numberOfSenders;
     int _numberOfReceivers;
     int _rank;
     int _nSlicers;
     int _nSegmentForceDetectors;
     int _nThreads;
     ThreadUserData* _threadUserData;
     Decomposition** _decomposition;
     TouchSpace* _detectionSegmentForceSpace;
     NeuronPartitioner* _neuronPartitioner;
     bool _writeToFile;
     Params* _params;

     // Receive Phase 0: ALLTOALL
     int *_segmentsPerSender; 		    // recvbuf
     MPI_Datatype _typeInt;                 // recvtype
        
     // Receive Phase 1: ALLTOALLW/V
     Capsule *_segmentData;
     int* _segmentCounts;         // recvcount
     int* _segmentDispls;         // recvDispls
     MPI_Datatype* _typeSegments;  // recvtype(s)
     int _segmentDataSize;
     int _previousSegBufSize;
     
     // Send Phase 0: ALLTOALL
     int *_segmentForcesPerReceiver;
     
     // Send Phase 1: ALLTOALLV
     SegmentForce* _segmentForceArray;
     int* _segmentForceCounts;
     int* _segmentForceDispls;
     MPI_Datatype _typeSegmentForce;
     int _segmentForceDataSize;

     // Send Phase 2: ALLREDUCESUM
     double _E0; // sendbuf
     int _one; 	 // sendcount
     MPI_Datatype _typeDouble; // senttype

     bool _updateCoveredSegments;
     Capsule* _coveredSegments;
     int _coveredSegsCount;

     static SegmentDescriptor _segmentDescriptor;
     static SegmentForce _segmentForce;
};

#endif
