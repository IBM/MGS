// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef LENSTISSUESLICER_H
#define LENSTISSUESLICER_H

#include <mpi.h>
#include <cassert>
#include <vector>
#include "TissueSlicer.h"
#include "SegmentDescriptor.h"
#include "TouchVector.h"
#include "Decomposition.h"

#include "MaxComputeOrder.h"
#include "TissueContext.h"

class TissueContext;
class Params;

class LENSTissueSlicer : public TissueSlicer
{
   public:
     LENSTissueSlicer(const int rank, 
		      const int nSlicers,
		      const int nTouchDetectors, 
		      TissueContext* tissueContext, 
		      Params* params=0);
     virtual ~LENSTissueSlicer();

   private:
     virtual void sliceAllNeurons();
     virtual void writeBuff(int i, int j, int& writePos);
     virtual void* getSendBuff();

     bool _sliced;
     TissueContext* _tissueContext;
     SegmentDescriptor _segmentDescriptor;
     //TUAN NOTE: this is currently not being used
     //  what is it was designed for?
     std::map<int, std::vector<long int> > _foreignTouchCapsuleMap;
};

#endif

