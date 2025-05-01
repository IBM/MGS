// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef ToroidalRadialSampler_H
#define ToroidalRadialSampler_H
#include "Mgs.h"

#include "CG_ToroidalRadialSamplerBase.h"
#include "LensContext.h"
#include "ParameterSet.h"
#include "NodeDescriptor.h"
#include <memory>
#include <vector>

class ToroidalRadialSampler : public CG_ToroidalRadialSamplerBase
{
   public:
      void userInitialize(LensContext* CG_c, float& radius);
      void userExecute(LensContext* CG_c);
      ToroidalRadialSampler();
      virtual ~ToroidalRadialSampler();
      virtual void duplicate(std::unique_ptr<ToroidalRadialSampler>&& dup) const;
      virtual void duplicate(std::unique_ptr<Functor>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_ToroidalRadialSamplerBase>&& dup) const;

   private:
      NodeDescriptor *_refNode;
      std::vector<NodeDescriptor*> _nodes;
      int _currentNode;
      int _nbrNodes;
      std::vector<int> _refcoords;
      std::vector<int> _nodeSetSize;
};

#endif
