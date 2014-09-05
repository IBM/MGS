// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// ================================================================

#ifndef TOUCHDETECTTISSUESLICER_H
#define TOUCHDETECTTISSUESLICER_H

#include <mpi.h>
#include "TissueSlicer.h"

class Tissue;
class Decomposition;
class Segment;
class Params;
class Capsule;
class TissueContext;
class TouchSpace;

class TouchDetectTissueSlicer : public TissueSlicer
{
   public:
     TouchDetectTissueSlicer(const int rank, const int nSlicers, const int nTouchDetectors,
			  Tissue* tissue, Decomposition** decomposition, TissueContext* tissueContext,
			  Params* params, const int maxComputeOrder);
     virtual ~TouchDetectTissueSlicer();
     void sendLostDaughters(bool sendLostDaughters) {_sendLostDaughters=sendLostDaughters;}
     void addCutPointJunctions(bool addCutPointJunctions) {_addCutPointJunctions=addCutPointJunctions;}
     void addTolerance(double tolerance) {_tolerance+=tolerance;}

   protected:
     virtual void sliceAllNeurons();
     virtual void writeBuff(int i, int j, int& writePos);
     virtual void* getSendBuff();

   private:
     int _maxComputeOrder;
     Segment* _segs; // also sendBuf
     bool _sendLostDaughters;
     bool _addCutPointJunctions;
     TissueContext* _tissueContext;
     double _tolerance;
};

#endif

