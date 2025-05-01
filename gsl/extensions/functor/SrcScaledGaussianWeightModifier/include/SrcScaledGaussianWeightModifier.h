// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef SrcScaledGaussianWeightModifier_H
#define SrcScaledGaussianWeightModifier_H
#include "Mgs.h"

#include "CG_SrcScaledGaussianWeightModifierBase.h"
#include "LensContext.h"
#include "ParameterSet.h"
#include <memory>

class SrcScaledGaussianWeightModifier : public CG_SrcScaledGaussianWeightModifierBase
{
   public:
      void userInitialize(LensContext* CG_c, Functor*& f, float& sigma, float& max);
      std::unique_ptr<ParameterSet> userExecute(LensContext* CG_c);
      SrcScaledGaussianWeightModifier();
      virtual ~SrcScaledGaussianWeightModifier();
      virtual void duplicate(std::unique_ptr<SrcScaledGaussianWeightModifier>&& dup) const;
      virtual void duplicate(std::unique_ptr<Functor>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_SrcScaledGaussianWeightModifierBase>&& dup) const;

      float _sigma;
      float _max;
};

#endif
