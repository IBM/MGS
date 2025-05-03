// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef DstScaledContractedGaussianWeightModifier_H
#define DstScaledContractedGaussianWeightModifier_H

#include "Mgs.h"
#include "CG_DstScaledContractedGaussianWeightModifierBase.h"
#include "GslContext.h"
#include "ParameterSet.h"
#include <memory>

class DstScaledContractedGaussianWeightModifier : public CG_DstScaledContractedGaussianWeightModifierBase
{
   public:
      void userInitialize(GslContext* CG_c, Functor*& f, float& sigma, float& max, float& contract);
      std::unique_ptr<ParameterSet> userExecute(GslContext* CG_c);
      DstScaledContractedGaussianWeightModifier();
      virtual ~DstScaledContractedGaussianWeightModifier();
      virtual void duplicate(std::unique_ptr<DstScaledContractedGaussianWeightModifier>&& dup) const;
      virtual void duplicate(std::unique_ptr<Functor>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_DstScaledContractedGaussianWeightModifierBase>&& dup) const;

      float _sigma, _max, _contract;
};

#endif
