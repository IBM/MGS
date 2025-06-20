// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef DstDimensionConstrainedSampler_H
#define DstDimensionConstrainedSampler_H

#include "Mgs.h"
#include "CG_DstDimensionConstrainedSamplerBase.h"
#include "GslContext.h"
#include <memory>
#include <vector>

class VolumeOdometer;
class NodeDescriptor;

class DstDimensionConstrainedSampler : public CG_DstDimensionConstrainedSamplerBase
{
   public:
      void userInitialize(GslContext* CG_c, int& constrainedDim);
      void userExecute(GslContext* CG_c);
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
