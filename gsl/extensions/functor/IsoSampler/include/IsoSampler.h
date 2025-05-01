// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef IsoSampler_H
#define IsoSampler_H

#include "Mgs.h"
#include "CG_IsoSamplerBase.h"
#include "LensContext.h"
#include <memory>
#include <vector>

class VolumeOdometer;
class NodeDescriptor;

class IsoSampler : public CG_IsoSamplerBase
{
   public:
      void userInitialize(LensContext* CG_c);
      void userExecute(LensContext* CG_c);
      IsoSampler();
      virtual ~IsoSampler();
      virtual void duplicate(std::unique_ptr<IsoSampler>&& dup) const;
      virtual void duplicate(std::unique_ptr<Functor>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_IsoSamplerBase>&& dup) const;

   private:
      bool _done;
      int _nbrNodes;
      std::vector<NodeDescriptor*> _srcNodes;
      std::vector<NodeDescriptor*> _dstNodes;
      int _nodeIndex;
};

#endif
