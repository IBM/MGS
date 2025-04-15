// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef TOUCHANALYZER_H
#define TOUCHANALYZER_H

#define TOUCH_ANALYZER_MERGE_PHASES 1
#define TOUCH_ANALYZER_FINALIZE_PHASES 2

#define BROADCAST_ROOT_TABLE 0

#include <mpi.h>
#include <vector>

#include "Receiver.h"
#include "Sender.h"
#include "Touch.h"
#include "TouchTable.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
#include <map>

class TouchDetector;
class Tissue;
class TouchFilter;
class TouchAnalysis;
class TouchTable;

class TouchAnalyzer : public Sender, public Receiver
{
   public:
     TouchAnalyzer(int rank, std::string experimentName, const int nSlicers, const int nTouchDetectors, 
		   TouchDetector* touchDetector, Tissue* tissue, int maxIterations, 
		   TouchFilter* touchFilter, bool writeToFile, bool output);      
     virtual ~TouchAnalyzer();
     
     void addTouchAnalysis(TouchAnalysis* touchAnalysis);
     TouchTable* addTouchTable(TouchTable* touchTable);
     void evaluateTouch(Touch& t);
     void outputTables(unsigned int iteration);
     void reset();
     virtual void analyze(unsigned int iteration);
     virtual bool isDone() {return _done;}

     int getRank() {return _rank;}

     void prepareToSend(int sendCycle, int sendPhase, CommunicatorFunction& funPtrRef);
     void* getSendbuf(int sendCycle, int sendPhase);
     int* getSendcounts(int sendCycle, int sendPhase);
     int* getSdispls(int sendCycle, int sendPhase);
     MPI_Datatype* getSendtypes(int sendCycle, int sendPhase);
     int getNumberOfSendCycles() {return _numberOfTables*(1+BROADCAST_ROOT_TABLE);}
     int getNumberOfSendPhasesPerCycle(int sendCycle);
     void finalizeReceive(int receiveCycle, int receivePhase);
     int getNumberOfReceivers() {return _numberOfReceivers;}
     
     void prepareToReceive(int receiveCycle, int receivePhase, CommunicatorFunction& funPtrRef);
     void* getRecvbuf(int receiveCycle, int receivePhase);
     int* getRecvcounts(int receiveCycle, int receivePhase);
     int* getRdispls(int receiveCycle, int receivePhase);
     MPI_Datatype* getRecvtypes(int receiveCycle, int receivePhase);
     int* getRecvSizes(int receiveCycle, int receivePhase);
     int getNumberOfReceiveCycles() {return _numberOfTables*(1+BROADCAST_ROOT_TABLE);}
     int getNumberOfReceivePhasesPerCycle(int receiveCycle);
     void mergeWithSendBuf(int index, int count, int sendCycle, int sendPhase);
     int getNumberOfSenders() {return _numberOfSenders;}
     void confirmTouchCounts(long long count);     
 
 protected:
     Tissue* _tissue;

   private:

     std::string _experimentName;
     int _rank;
     int _nSlicers;
     int _nTouchDetectors;
     TouchDetector* _touchDetector;
     
     bool _done;
     int _numberOfSenders;
     int _numberOfReceivers;
		 
     std::vector<TouchAnalysis*> _touchAnalyses;
     std::vector<TouchTable*> _touchTables;
     int _numberOfTables;

     // Send Phase 0: number of table entries
     int _tableSize;           // sendbuf
     MPI_Datatype _typeInt;    // send and receive type

     // Send Phase 1: table entries
     std::vector<TouchTableEntry*> _tableRecvBufs;
     std::vector<TouchTableEntry*>::iterator _tableRecvBufsIter;

     std::vector<TouchTableEntry*> _tableSendBufs;
     std::vector<TouchTableEntry*>::iterator _tableSendBufsIter;

     TouchTableEntry* _tableEntriesSendBuf;// sendbuf 
     TouchTableEntry* _tableEntriesRecvBuf;// receivebuf

     int _sendBufSize, _recvBufSize;
     int _one;
     int _tableBufSize;

     MPI_Datatype _typeTouchTableEntry;// send and receive type

     std::map<double, std::map<double, long> >::iterator _mergeIterator1;
     std::map<double, long>::iterator _mergeIterator2;
     bool _mergeComplete;

     TouchFilter* _touchFilter;
     int _maxIterations;
     bool _writeToFile;
     bool _output;
};

#endif



