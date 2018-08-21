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

#include <mpi.h>
#include "History.h"
#include "string.h"
#include <algorithm>

History::History(FILE* hfile, int fid, int elementLength) :
  _history(0), _historyIter(0), _iterations(0), _iterationsIter(0), _iterationsEnd(0), 
  _historyFile(hfile), _fid(fid), _elementLength(elementLength)
{
  _history = _historyIter = new double[HISTORY_BUFF_SIZE*_elementLength];
  _iterations = _iterationsIter = new int[HISTORY_BUFF_SIZE];
  _iterationsEnd = _iterations + HISTORY_BUFF_SIZE;
}

void History::flush()
{
  fwrite(&_fid, sizeof(int), 1, _historyFile);
  int count = _iterationsIter-_iterations;
  fwrite(&count, sizeof(int), 1, _historyFile);
  fwrite(_iterations, sizeof(int), count, _historyFile);
  fwrite(_history, sizeof(double), count*_elementLength, _historyFile);
  _historyIter = _history;	
	_iterationsIter = _iterations;
}

void History::add(double* t, int iteration)
{
  *_iterationsIter = iteration;
  std::copy(t, t+ _elementLength, _historyIter);
//  memcpy(_historyIter, t, sizeof(double)*_elementLength);
  ++_iterationsIter;
  _historyIter += _elementLength;
  if (_iterationsIter == _iterationsEnd) flush();
}

History::~History()
{
  if (_historyIter-_history>0) flush();
  delete _history;
	delete _iterations;
}
