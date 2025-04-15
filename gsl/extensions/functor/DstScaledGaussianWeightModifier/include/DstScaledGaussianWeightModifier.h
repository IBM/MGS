// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef DstScaledGaussianWeightModifier_H
#define DstScaledGaussianWeightModifier_H
#include "Lens.h"

#include "CG_DstScaledGaussianWeightModifierBase.h"
#include "LensContext.h"
#include "ParameterSet.h"
#include <memory>

class DstScaledGaussianWeightModifier : public CG_DstScaledGaussianWeightModifierBase
{
   public:
      void userInitialize(LensContext* CG_c, Functor*& f, float& sigma, float& max);
      std::unique_ptr<ParameterSet> userExecute(LensContext* CG_c);
      DstScaledGaussianWeightModifier();
      virtual ~DstScaledGaussianWeightModifier();
      virtual void duplicate(std::unique_ptr<DstScaledGaussianWeightModifier>&& dup) const;
      virtual void duplicate(std::unique_ptr<Functor>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_DstScaledGaussianWeightModifierBase>&& dup) const;

      float _sigma;
      float _max;
};

#endif
