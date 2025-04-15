// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef HISTORY_H
#define HISTORY_H

#include <mpi.h>
#include <stdio.h>

#define HISTORY_BUFF_SIZE 1000

class History
{
 public:
  History(FILE*, int fid, int elementLength);
  void add(double* t, int iteration);
  double const *getHistory() {return _history;}
  double const *getLast() {return _historyIter-3;}
  void flush();
  ~History();

 private:
  double* _history;
  double* _historyIter;
  int* _iterations;
  int* _iterationsIter;
  int* _iterationsEnd;
  FILE* _historyFile;
  int _fid;
  int _elementLength;
};

#endif
