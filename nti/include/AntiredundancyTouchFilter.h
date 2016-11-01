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

