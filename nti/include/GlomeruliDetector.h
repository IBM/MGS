// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef GLOMERULIDETECTOR_H
#define GLOMERULIDETECTOR_H

#include <mpi.h>

class TouchVector;

class GlomeruliDetector
{
  public:	
 
    virtual void findGlomeruli(TouchVector*)=0;
    virtual double getGlomeruliSpacing()=0;
    virtual ~GlomeruliDetector() {}
};

#endif
