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
