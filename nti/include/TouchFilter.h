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
