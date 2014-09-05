#ifndef SrcDimensionConstrainedSampler_H
#define SrcDimensionConstrainedSampler_H

#include "Lens.h"
#include "CG_SrcDimensionConstrainedSamplerBase.h"
#include "LensContext.h"
#include <memory>
#include <vector>

class VolumeOdometer;
class NodeDescriptor;

class SrcDimensionConstrainedSampler : public CG_SrcDimensionConstrainedSamplerBase
{
   public:
      void userInitialize(LensContext* CG_c, int& constrainedDim);
      void userExecute(LensContext* CG_c);
      SrcDimensionConstrainedSampler();
      virtual ~SrcDimensionConstrainedSampler();
      virtual void duplicate(std::auto_ptr<SrcDimensionConstrainedSampler>& dup) const;
      virtual void duplicate(std::auto_ptr<Functor>& dup) const;
      virtual void duplicate(std::auto_ptr<CG_SrcDimensionConstrainedSamplerBase>& dup) const;

   private:
      int _constrainedSrcDim;
      int _shortDstDim;
      int _currentConstrainedDimOffsetSrc;
      int _beginConstrainedDimSrc;
      int _endConstrainedDimSrc;
      bool _done;
      int _nbrNodesSrc;
      int _nbrNodesDst;
      std::vector<NodeDescriptor*> _srcNodes;
      std::vector<NodeDescriptor*> _dstNodes;
      int _dstNodeIndex;
      bool _next;
};

#endif
