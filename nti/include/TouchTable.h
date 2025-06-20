// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef TOUCHTABLE_H
#define TOUCHTABLE_H

#include <mpi.h>
#include <map>
#include <vector>

#include "SegmentDescriptor.h"
#include <limits.h>

class TouchTableEntry
{
 public:
  TouchTableEntry() : key1(0), key2(0), count(0) {}
  ~TouchTableEntry() {}

  double key1;
  double key2;
  long count;
};

class TouchTable
{
 public:
  TouchTable(std::vector<SegmentDescriptor::SegmentKeyData> const & maskVector);
  ~TouchTable();
  void evaluateTouch(double segKey1, double segKey2);
  void writeToFile(int tableNumber, int iterationNumber, std::string experimentName);
  void outputTable(int tableNumber, int iterationNumber, std::string experimentName);
  void setOutput(bool output) {_output=output;}
  void reset();
  int size();
  int getTouchCount();
  unsigned long long getMask() {return _mask;}
  void getEntries(TouchTableEntry* outputBuffer);
  void getEntries(TouchTableEntry* outputBuffer, 
		  int numberOfEntries,
		  std::map<double, std::map<double, long> >::iterator& mapIter1,
		  std::map<double, long>::iterator& mapIter2,
		  bool& complete);
  void addEntries(TouchTableEntry* buffer, TouchTableEntry* bufferEnd); 

 private:
  std::map<double, std::map<double, long> > _touchTable;
  unsigned long long _mask;
  SegmentDescriptor _segmentDescriptor;
  bool _output;
};
 
#endif

