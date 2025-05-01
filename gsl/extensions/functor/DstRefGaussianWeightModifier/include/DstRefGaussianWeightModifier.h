// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef DstRefGaussianWeightModifier_H
#define DstRefGaussianWeightModifier_H
#include "Mgs.h"

#include "CG_DstRefGaussianWeightModifierBase.h"
#include "LensContext.h"
#include "ParameterSet.h"
#include <memory>

class DstRefGaussianWeightModifier : public CG_DstRefGaussianWeightModifierBase
{
   public:
      void userInitialize(LensContext* CG_c, Functor*& f, float& sigma, float& max, int& wrapDistance);
      std::unique_ptr<ParameterSet> userExecute(LensContext* CG_c);
      DstRefGaussianWeightModifier();
      virtual ~DstRefGaussianWeightModifier();
      virtual void duplicate(std::unique_ptr<DstRefGaussianWeightModifier>&& dup) const;
      virtual void duplicate(std::unique_ptr<Functor>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_DstRefGaussianWeightModifierBase>&& dup) const;

      float _sigma;
      float _max;
      int _wrapDistance;
};

#endif
