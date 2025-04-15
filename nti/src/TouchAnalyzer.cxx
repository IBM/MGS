// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "TouchAnalyzer.h"
#include "TouchDetector.h"
#include "Tissue.h"
#include "TouchFilter.h"
#include "TouchAnalysis.h"
#include "TouchTable.h"
#include "Rotation.h"
#include "Translation.h"
#include "Utilities.h"
#include "BuffFactor.h"

#include <list>
#include <cassert>

//#define RECORD_TOUCHES

TouchAnalyzer::TouchAnalyzer(
			     int rank, 
			     std::string experimentName,
			     const int nSlicers, 
			     const int nTouchDetectors,
			     TouchDetector* touchDetector,
			     Tissue* tissue,
			     int maxIterations,
			     TouchFilter* touchFilter,
			     bool writeToFile,
			     bool output)
  : _tissue(tissue),
    _rank(rank),
    _experimentName(experimentName),
    _nSlicers(nSlicers),
    _nTouchDetectors(nTouchDetectors),
    _touchDetector(touchDetector),
    _done(false),
    _numberOfSenders(0),
    _numberOfReceivers(0),
    _numberOfTables(0),
    _tableSize(0),
    _typeInt(MPI_INT),
    _tableEntriesSendBuf(0),
    _tableEntriesRecvBuf(0),
    _sendBufSize(1),
    _recvBufSize(1),
    _one(1),
    _tableBufSize(MERGE_BUFF_SIZE),
    _mergeComplete(true),
    _touchFilter(touchFilter),
    _maxIterations(maxIterations),
    _writeToFile(writeToFile),
    _output(output)
{
  _touchDetector->setTouchAnalyzer(this);
  if (_touchFilter) {
    _touchFilter->setTouchAnalyzer(this);
  }
  if (_maxIterations==1) {
    _done = true;
#ifdef RECORD_TOUCHES
    if (_writeToFile) _touchDetector->writeToFile(_experimentName);
#endif
  }
  _numberOfReceivers = _numberOfSenders = (nSlicers>nTouchDetectors)?nSlicers:nTouchDetectors;
  
  TouchTableEntry tableEntry;

  Datatype datatype(4, &tableEntry);
  datatype.set(0, MPI_LB, 0);
  datatype.set(1, MPI_DOUBLE, 2, &tableEntry.key1);
  datatype.set(2, MPI_LONG, 1, &tableEntry.count);
  datatype.set(3, MPI_UB, sizeof(TouchTableEntry));
  _typeTouchTableEntry = datatype.commit();
  _tableEntriesSendBuf = new TouchTableEntry[_sendBufSize];
  _tableEntriesRecvBuf = new TouchTableEntry[_recvBufSize];
}

TouchAnalyzer::~TouchAnalyzer()
{
  delete [] _tableEntriesSendBuf;
  delete [] _tableEntriesRecvBuf;
  for (int i=0; i<_tableRecvBufs.size(); ++i) delete [] _tableRecvBufs[i];
  for (int i=0; i<_tableSendBufs.size(); ++i) delete [] _tableSendBufs[i];
  std::vector<TouchAnalysis*>::iterator iter2;
  std::vector<TouchAnalysis*>::iterator end2=_touchAnalyses.end();
  for (iter2=_touchAnalyses.begin(); iter2 != end2; ++iter2) delete (*iter2);
  std::vector<TouchTable*>::iterator iter;
  std::vector<TouchTable*>::iterator end=_touchTables.end();
  for (iter=_touchTables.begin(); iter != end; ++iter) delete (*iter);
}


void TouchAnalyzer::addTouchAnalysis(TouchAnalysis* touchAnalysis)
{
  _touchAnalyses.push_back(touchAnalysis);
  std::vector<TouchTable*> & addedTouchTables = touchAnalysis->getTouchTables();
  std::vector<TouchTable*>::iterator iter, end=addedTouchTables.end();
  for (iter=addedTouchTables.begin(); iter != end; ++iter) {
    TouchTable* t1 = (*iter);
    TouchTable* t2 = addTouchTable(t1);
    if (t1 != t2) {
      delete t1;
      (*iter) = t2;
    }
  }
}

TouchTable* TouchAnalyzer::addTouchTable(TouchTable* newOne)
{
  TouchTable* rval = newOne;
  std::vector<TouchTable*>::iterator iter;
  std::vector<TouchTable*>::iterator end=_touchTables.end();
  for (iter=_touchTables.begin(); iter != end; ++iter) {
    TouchTable* oldOne = (*iter);
    if (newOne->getMask()==oldOne->getMask()) {
      rval = oldOne;
      break;
    }
  }
  if (rval==newOne) {
    _touchTables.push_back(newOne);
    ++_numberOfTables;
  }
  return rval;
}

void TouchAnalyzer::evaluateTouch(Touch& t)
{
  double segKey1=t.getKey1();
  double segKey2=t.getKey2();
  std::vector<TouchTable*>::iterator iter;
  std::vector<TouchTable*>::iterator end=_touchTables.end();
  for (iter=_touchTables.begin(); iter != end; ++iter) {
    (*iter)->evaluateTouch(segKey1, segKey2);
  }
}

void TouchAnalyzer::confirmTouchCounts(long long count)
{
  std::vector<TouchTable*>::iterator iter;
  std::vector<TouchTable*>::iterator end=_touchTables.end();
  for (iter=_touchTables.begin(); iter != end; ++iter) {
    assert((*iter)->getTouchCount()==count);
  }
}

void TouchAnalyzer::reset()
{
  std::vector<TouchTable*>::iterator iter;
  std::vector<TouchTable*>::iterator end=_touchTables.end();
  for (iter=_touchTables.begin(); iter != end; ++iter) {
    (*iter)->reset();
  }
}

void TouchAnalyzer::outputTables(unsigned int iteration)
{
  if (_rank==0) {
    std::vector<TouchTable*>::iterator iter, end = _touchTables.end();
    int n=0;
    for (iter = _touchTables.begin(); iter != end; ++iter, ++n) {
      if (_writeToFile) (*iter)->writeToFile(n, iteration, _experimentName);
      if (_output) (*iter)->outputTable(n, iteration, _experimentName);
    }
  }
}

void TouchAnalyzer::analyze(unsigned int iteration)
{
  outputTables(iteration);
	
  std::list<Translation> translations;
  std::list<Rotation> rotations;
  std::vector<TouchAnalysis*>::iterator iter, end = _touchAnalyses.end();
  _done = true;
  for (iter = _touchAnalyses.begin(); iter != end; ++iter)
    _done = _done && (*iter)->analyze(translations, rotations);
  if (iteration==_maxIterations-1) _done=true;
#ifdef RECORD_TOUCHES
  if (_done && _writeToFile) _touchDetector->writeToFile(_experimentName);
#endif
  translations.sort();

  std::list<Translation>::iterator currentTrans, transIter=translations.begin(), transEnd=translations.end();

  while (transIter!=transEnd) {
    currentTrans=transIter;
    ++transIter;
    while (transIter != transEnd && (*currentTrans)==(*transIter)) {				
      (*currentTrans)+=(*transIter);
      ++transIter;
    }
  }
  translations.unique();
  transEnd = translations.end();
	
  rotations.sort();
  std::list<Rotation>::iterator currentRot, rotIter=rotations.begin(), rotEnd=rotations.end();

  while (rotIter!=rotEnd) {
    currentRot=rotIter;
    ++rotIter;
    while (rotIter != rotEnd && (*currentRot)==(*rotIter)) {				
      (*currentRot)+=(*rotIter);
      ++rotIter;
    }
  }

  rotations.unique();
  rotEnd = rotations.end();


  for (transIter=translations.begin(); transIter != transEnd; ++transIter) {
    _tissue->translateNeuron(transIter->getIndex(),transIter->getTranslation(), iteration);
  }	

  for (rotIter=rotations.begin(); rotIter != rotEnd; ++rotIter) {
    _tissue->rotateNeuronY(rotIter->getIndex(),rotIter->getRotation(), iteration);
  }
  reset();
}

// SEND Methods

void TouchAnalyzer::prepareToSend(int sendCycle, int sendPhase, CommunicatorFunction& funPtrRef)
{
  if (sendCycle==0) {
    if (_touchFilter) _touchFilter->filterTouches();
  }

  if (sendCycle<_numberOfTables) {
    assert(sendPhase==0);
    _mergeComplete=true;
    funPtrRef = &Communicator::tableMerge;
  }
  else {
    assert(sendPhase<=1);
    if (_rank==0) {
      if (sendPhase==0) {
	int tableIndex=sendCycle-_numberOfTables;
	_tableSize = _touchTables[tableIndex]->size();
      }
      else {
	int tableIndex=sendCycle-_numberOfTables;
	_tableSize = _touchTables[tableIndex]->size();
	if (_sendBufSize<_tableSize) {
	  _sendBufSize = getBuffAllocationSize(_tableSize);
	  delete [] _tableEntriesSendBuf;
	  _tableEntriesSendBuf = new TouchTableEntry[_sendBufSize];
	}
	_touchTables[tableIndex]->getEntries(_tableEntriesSendBuf);
      }
    }
    funPtrRef = &Communicator::bcast;
  }
}

void* TouchAnalyzer::getSendbuf(int sendCycle, int sendPhase)
{  
  void* rval = 0;
  if (sendCycle<_numberOfTables) {
    TouchTableEntry* tte = 0;
    int tableIndex=sendCycle;
    if (_tableSendBufs.size()>sendPhase)
      tte = _tableSendBufs[sendPhase];
    else {
      tte = new TouchTableEntry[_tableBufSize];
      _tableSendBufs.push_back(tte);	
      assert(_tableSendBufs.size()>sendPhase);
    }
    _touchTables[tableIndex]->getEntries(tte,
					 _tableBufSize,
					 _mergeIterator1,
					 _mergeIterator2,
					 _mergeComplete);
    rval=tte;
 }
  else {
    assert(_rank==0);
    assert(sendPhase<=1);
    if (sendPhase==0) rval = &_tableSize;
    else rval = _tableEntriesSendBuf;
  }
  return rval;
}

int* TouchAnalyzer::getSendcounts(int sendCycle, int sendPhase)
{
  int* rval = 0;
  if (sendCycle<_numberOfTables) {
    assert(sendPhase==0);
    int tableIndex=sendCycle;
    _tableSize = _touchTables[tableIndex]->size();
    rval = &_tableSize;
  }
  else {
    assert(_rank==0);
    assert(sendPhase<=1);
    if (sendPhase==0)
      rval = &_one;
    else if (sendPhase==1) {
      rval = &_tableSize;
    }
  }
  return rval;
}

int* TouchAnalyzer::getSdispls(int sendCycle, int sendPhase)
{
  int* rval = 0;
  assert(0);
  return rval;
}

MPI_Datatype* TouchAnalyzer::getSendtypes(int sendCycle, int sendPhase)
{
  MPI_Datatype* rval = 0;
  if (sendCycle<_numberOfTables) {
    assert(sendPhase==0);
    rval = &_typeTouchTableEntry;
  }
  else {
    assert(_rank==0);
    assert(sendPhase<=1);
    if (sendPhase==0)
      rval = &_typeInt;
    else if (sendPhase==1)
      rval = &_typeTouchTableEntry;
   }
  return rval;
}

int TouchAnalyzer::getNumberOfSendPhasesPerCycle(int sendCycle)
{
  int rval = 0;
  if (sendCycle<_numberOfTables) rval=TOUCH_ANALYZER_MERGE_PHASES;
  else if (sendCycle>=_numberOfTables) rval=TOUCH_ANALYZER_FINALIZE_PHASES;
  return rval;
}

void TouchAnalyzer::mergeWithSendBuf(int index, int count, int sendCycle, int sendPhase)
{
  assert(sendCycle<_numberOfTables);
  int tableIndex=sendCycle;
  TouchTableEntry* recvbufPtr = _tableRecvBufs[index];
  _touchTables[tableIndex]->addEntries(recvbufPtr, recvbufPtr+count);
  _tableSize = _touchTables[tableIndex]->size();
}

// RECEIVE Methods

void TouchAnalyzer::prepareToReceive(int receiveCycle, int receivePhase, CommunicatorFunction& funPtrRef)
{
  if (receiveCycle<_numberOfTables) {
    assert(receivePhase==0);
    funPtrRef = &Communicator::tableMerge;
  }
  else { 
    assert (receivePhase<=1);
    if (receivePhase==1 && _rank!=0) {
      int tableIndex=receiveCycle-_numberOfTables;
      if (_recvBufSize<_tableSize) {
	_recvBufSize=getBuffAllocationSize(_tableSize);
	delete [] _tableEntriesRecvBuf;
	_tableEntriesRecvBuf = new TouchTableEntry[_recvBufSize];
      }
    }
    funPtrRef = &Communicator::bcast;
  }
}

void* TouchAnalyzer::getRecvbuf(int receiveCycle, int receivePhase)
{
  void* rval = 0;
  if (receiveCycle<_numberOfTables) {
    assert(receivePhase==0);    
    if (receiveCycle==0) {
      TouchTableEntry* tte = new TouchTableEntry[_tableBufSize];
      _tableRecvBufs.push_back(tte);
      _tableRecvBufsIter=_tableRecvBufs.begin();
      rval=tte;
    }
    else {
      rval = *_tableRecvBufsIter;
      if (++_tableRecvBufsIter==_tableRecvBufs.end()) _tableRecvBufsIter=_tableRecvBufs.begin();
    }
  }
  else {
    assert(_rank!=0);
    assert (receivePhase<=1);
    if (receivePhase==0)
      rval = &_tableSize;
    else if (receivePhase==1) 
      rval = _tableEntriesRecvBuf;
  }
  return rval;
}

int* TouchAnalyzer::getRecvcounts(int receiveCycle, int receivePhase)
{
  int* rval = 0;
  if (receiveCycle<_numberOfTables) {
    assert(receivePhase==0);
    rval = &_tableBufSize;
  }
  else {
    assert(_rank!=0);
    assert (receivePhase<=1);
    if (receivePhase==0)
      rval = &_one;
    else if (receivePhase==1)
      rval = &_tableSize;
  }
  return rval;
}

int* TouchAnalyzer::getRdispls(int receiveCycle, int receivePhase)
{
  int* rval = 0;
  assert(0);
  return rval;
}

MPI_Datatype* TouchAnalyzer::getRecvtypes(int receiveCycle, int receivePhase)
{
  MPI_Datatype* rval = 0;
  if (receiveCycle<_numberOfTables) {
    assert(receivePhase==0);
    rval = &_typeTouchTableEntry;
  }
  else {
    assert(_rank!=0);
    assert (receivePhase<=1);
    if (receivePhase==0)
      rval = &_typeInt;
    else if (receivePhase==1)
      rval = &_typeTouchTableEntry;
  }
  return rval;
}

int TouchAnalyzer::getNumberOfReceivePhasesPerCycle(int receiveCycle)
{
  int rval = 0;
  if (receiveCycle<_numberOfTables) rval=TOUCH_ANALYZER_MERGE_PHASES;
  else if (receiveCycle>=_numberOfTables) rval=TOUCH_ANALYZER_FINALIZE_PHASES;
  return rval;
}

void TouchAnalyzer::finalizeReceive(int receiveCycle, int receivePhase)
{
  if (receiveCycle>=_numberOfTables && receivePhase==1 && _rank!=0) {
    int tableIndex=receiveCycle-_numberOfTables;
    _touchTables[tableIndex]->reset();
    _touchTables[tableIndex]->addEntries(_tableEntriesRecvBuf, _tableEntriesRecvBuf+_tableSize);
  }  
}
