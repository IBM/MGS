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

#ifndef LENSTISSUESLICER_H
#define LENSTISSUESLICER_H

#include <mpi.h>
#include <cassert>
#include <vector>
#include "TissueSlicer.h"
#include "SegmentDescriptor.h"
#include "TouchVector.h"
#include "Decomposition.h"

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
     std::map<int, std::vector<long int> > _foreignTouchCapsuleMap;
};

#endif

