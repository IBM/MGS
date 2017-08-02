// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// ================================================================

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
