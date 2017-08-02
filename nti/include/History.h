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
