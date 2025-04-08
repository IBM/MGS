// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef DstDimensionConstrainedSampler_H
#define DstDimensionConstrainedSampler_H

#include "Lens.h"
#include "CG_DstDimensionConstrainedSamplerBase.h"
#include "LensContext.h"
#include <memory>
#include <vector>

class VolumeOdometer;
class NodeDescriptor;

class DstDimensionConstrainedSampler : public CG_DstDimensionConstrainedSamplerBase
{
   public:
      void userInitialize(LensContext* CG_c, int& constrainedDim);
      void userExecute(LensContext* CG_c);
      DstDimensionConstrainedSampler();
      virtual ~DstDimensionConstrainedSampler();
      virtual void duplicate(std::unique_ptr<DstDimensionConstrainedSampler>&& dup) const;
      virtual void duplicate(std::unique_ptr<Functor>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_DstDimensionConstrainedSamplerBase>&& dup) const;

   private:
      int _constrainedDstDim;
      int _shortSrcDim;
      int _currentConstrainedDimOffsetDst;
      int _beginConstrainedDimDst;
      int _endConstrainedDimDst;
      bool _done;
      int _nbrNodesDst;
      int _nbrNodesSrc;
      std::vector<NodeDescriptor*> _dstNodes;
      std::vector<NodeDescriptor*> _srcNodes;
      int _srcNodeIndex;
      bool _next;
};

#endif
