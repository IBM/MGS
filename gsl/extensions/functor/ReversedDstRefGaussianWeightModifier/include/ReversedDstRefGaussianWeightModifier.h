// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef ReversedDstRefGaussianWeightModifier_H
#define ReversedDstRefGaussianWeightModifier_H
#include "Mgs.h"

#include "CG_ReversedDstRefGaussianWeightModifierBase.h"
#include "LensContext.h"
#include "ParameterSet.h"
#include <memory>

class ReversedDstRefGaussianWeightModifier : public CG_ReversedDstRefGaussianWeightModifierBase
{
   public:
      void userInitialize(LensContext* CG_c, Functor*& f, float& sigma, float& max, int& wrapDistance);
      std::unique_ptr<ParameterSet> userExecute(LensContext* CG_c);
      ReversedDstRefGaussianWeightModifier();
      virtual ~ReversedDstRefGaussianWeightModifier();
      virtual void duplicate(std::unique_ptr<ReversedDstRefGaussianWeightModifier>&& dup) const;
      virtual void duplicate(std::unique_ptr<Functor>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_ReversedDstRefGaussianWeightModifierBase>&& dup) const;

      float _sigma;
      float _max;
      int _wrapDistance;
};

#endif
