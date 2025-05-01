// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef SrcRefGaussianWeightModifier_H
#define SrcRefGaussianWeightModifier_H
#include "Mgs.h"

#include "CG_SrcRefGaussianWeightModifierBase.h"
#include "LensContext.h"
#include "ParameterSet.h"
#include <memory>

class SrcRefGaussianWeightModifier : public CG_SrcRefGaussianWeightModifierBase
{
   public:
      void userInitialize(LensContext* CG_c, Functor*& f, float& sigma, float& max, int& wrapDistance);
      std::unique_ptr<ParameterSet> userExecute(LensContext* CG_c);
      SrcRefGaussianWeightModifier();
      virtual ~SrcRefGaussianWeightModifier();
      virtual void duplicate(std::unique_ptr<SrcRefGaussianWeightModifier>&& dup) const;
      virtual void duplicate(std::unique_ptr<Functor>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_SrcRefGaussianWeightModifierBase>&& dup) const;

      float _sigma;
      float _max;
      int _wrapDistance;
};

#endif
