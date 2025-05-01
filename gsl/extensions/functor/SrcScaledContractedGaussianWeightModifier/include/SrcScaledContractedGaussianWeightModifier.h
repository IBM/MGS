// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef SrcScaledContractedGaussianWeightModifier_H
#define SrcScaledContractedGaussianWeightModifier_H

#include "Mgs.h"
#include "CG_SrcScaledContractedGaussianWeightModifierBase.h"
#include "LensContext.h"
#include "ParameterSet.h"
#include <memory>

class SrcScaledContractedGaussianWeightModifier : public CG_SrcScaledContractedGaussianWeightModifierBase
{
   public:
      void userInitialize(LensContext* CG_c, Functor*& f, float& sigma, float& max, float& contract);
      std::unique_ptr<ParameterSet> userExecute(LensContext* CG_c);
      SrcScaledContractedGaussianWeightModifier();
      virtual ~SrcScaledContractedGaussianWeightModifier();
      virtual void duplicate(std::unique_ptr<SrcScaledContractedGaussianWeightModifier>&& dup) const;
      virtual void duplicate(std::unique_ptr<Functor>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_SrcScaledContractedGaussianWeightModifierBase>&& dup) const;

      float _sigma, _max, _contract;
};

#endif
