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

