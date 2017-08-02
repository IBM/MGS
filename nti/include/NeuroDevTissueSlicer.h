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

#ifndef NEURODEVTISSUESLICER_H
#define NEURODEVTISSUESLICER_H

#include <mpi.h>

#include "TissueSlicer.h"
#include "SegmentForce.h"
#include "SegmentDescriptor.h"

#include <map>

class Tissue;
class Decomposition;
class Segment;
class SegmentSpace;
class Params;

class NeuroDevTissueSlicer : public TissueSlicer
{
   public:
     NeuroDevTissueSlicer(const int rank, const int nSlicers, const int nTouchDetectors, 
			Tissue* tissue, Decomposition** decomposition, 
			SegmentSpace* segmentSpace, Params* params, double& E);
     void resetSegmentSpace(SegmentSpace* segmentSpace) {_segmentSpace=segmentSpace;}
     virtual ~NeuroDevTissueSlicer();

   private:
     virtual void sliceAllNeurons();
     virtual void writeBuff(int i, int j, int& writePos);
     virtual void* getSendBuff();
     void computeTopologicalForces(int idx);

     double& _E;
     SegmentForce _segmentForce;
     SegmentSpace* _segmentSpace;
     Segment* _segs; // also sendBuf
     Segment* _segsEnd;
     int _segsSize;
     double* _forces;
     double* _forcesEnd;
     int _forcesSize;
     double* _R0a;
     double* _R0b;
     SegmentDescriptor _segmentDescriptor;
};

#endif

