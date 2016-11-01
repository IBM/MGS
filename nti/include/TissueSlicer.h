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

#ifndef TISSUESLICER_H
#define TISSUESLICER_H

#include <mpi.h>

#include "Sender.h"
#include <vector>
#include <cassert>

#define TISSUE_SLICER_SEND_PHASES 2

class Tissue;
class Decomposition;
class Params;


class TissueSlicer : public Sender
{
 public:
  TissueSlicer(const int rank,
	       const int nSlicers,
	       const int nTouchDetectors,
	       Tissue* tissue,
	       Decomposition** decomposition,
	       Params* params);
  virtual ~TissueSlicer();
  
  int getRank() {return _rank;}

  virtual void prepareToSend(int sendCycle, int sendPhase, CommunicatorFunction& funPtrRef);
  virtual void* getSendbuf(int sendCycle, int sendPhase);
  virtual int* getSendcounts(int sendCycle, int sendPhase);
  virtual int* getSdispls(int sendCycle, int sendPhase);
  virtual MPI_Datatype* getSendtypes(int sendCycle, int sendPhase);
  virtual int getNumberOfSendCycles() {return 1;}
  virtual int getNumberOfSendPhasesPerCycle(int sendCycle);
  virtual int getNumberOfReceivers() {return _numberOfReceivers;}
  virtual void mergeWithSendBuf(int index, int count, int sendCycle, int sendPhase) {assert(0);}

 protected:
  virtual void sliceAllNeurons()=0;
  virtual void* getSendBuff()=0;
  virtual void writeBuff(int i, int j, int& writePos)=0;

  virtual void initializePhase0();
  virtual void initializePhase1();
  
  int _rank;
  int _nSlicers;
  int _nTouchDetectors;  
  Tissue* _tissue;
  Decomposition** _decomposition;
  std::vector<long int>* _sliceSegmentIndices; // _sliceSegmentIndices[SLICE_NUMBER] 
  int _numberOfReceivers;
  int _maxNumSegs;
  int* _segmentBlockLengths;
  int* _segmentBlockDisplacements;

  // Send Phase 0: ALLTOALL : number of Segments's
  
  int *_segmentsPerSlice; // sendbuf
  int _numberOfInts;                    // sendcount
  MPI_Datatype _typeInt;                // sendtype

  // Send Phase 1: ALLTOALLV/W : Segments
  int *_segmentCounts;           // sendcount
  int *_segmentDispls;           // sendDispls
  MPI_Datatype *_typeSegments;   // sendtype (non-continuous set of typeSegment)
  MPI_Datatype _typeSegmentData; // primitive segment type
  double* _sendBuff;
  int _sendBuffSize;
  int _dataSize;

  Params* _params;
};

#endif

