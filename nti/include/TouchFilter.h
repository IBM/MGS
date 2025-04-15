// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef TOUCHFILTER_H
#define TOUCHFILTER_H

#include <mpi.h>

class TouchAnalyzer;

class TouchFilter
{
 public:	
  virtual void filterTouches() = 0;
  virtual void setTouchAnalyzer(TouchAnalyzer*) = 0;
};

#endif
