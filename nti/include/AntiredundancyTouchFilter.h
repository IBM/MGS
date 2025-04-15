// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef ANTIREDUNDANCYTOUCHFILTER_H
#define ANTIREDUNDANCYTOUCHFILTER_H

#include <mpi.h>
#include "TouchFilter.h"
#include "SegmentDescriptor.h"

#include <iostream>
#include <vector>
#include <math.h>
#include <list>

class Tissue;
class TouchAggregator;
class TouchAnalyzer;
class Touch;

class AntiredundancyTouchFilter : public TouchFilter

{
 public:
  AntiredundancyTouchFilter(TouchAggregator* touchAggregator, Tissue* tissue);
  virtual ~AntiredundancyTouchFilter();
			
  void filterTouches();
  void setTouchAnalyzer(TouchAnalyzer* touchAnalyzer);

 protected:
  TouchAnalyzer* _touchAnalyzer;

 private:
  void eliminateRedundantTouches(Touch* touchStart, Touch* touchEnd, int c);
  TouchAggregator* _touchAggregator;
  Tissue* _tissue;	
  unsigned long long _neuronMask;
  unsigned long long _branchMask;
  unsigned long long _segmentMask;
  SegmentDescriptor _segmentDescriptor;
};

#endif

