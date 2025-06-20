// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef PASSTHROUGHTOUCHFILTER_H
#define PASSTHROUGHTOUCHFILTER_H

#include <mpi.h>
#include "TouchFilter.h"
#include "SegmentDescriptor.h"

#include <iostream>
#include <vector>
#include <math.h>
#include <list>

class Tissue;
class TouchAggregator;
class Touch;
class TouchAnalyzer;

class PassThroughTouchFilter : public TouchFilter

{
 public:
  PassThroughTouchFilter(TouchAggregator* touchAggregator, Tissue* tissue);
  virtual ~PassThroughTouchFilter();

  virtual void filterTouches();
  void setTouchAnalyzer(TouchAnalyzer* touchAnalyzer);

 protected:
  TouchAnalyzer* _touchAnalyzer;

 private:
  TouchAggregator* _touchAggregator;
  Tissue* _tissue;
  SegmentDescriptor _segmentDescriptor;
};

#endif
