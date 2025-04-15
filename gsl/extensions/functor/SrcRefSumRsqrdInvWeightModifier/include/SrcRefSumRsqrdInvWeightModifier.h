// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef SrcRefSumRsqrdInvWeightModifier_H
#define SrcRefSumRsqrdInvWeightModifier_H
#include "Lens.h"

#include "CG_SrcRefSumRsqrdInvWeightModifierBase.h"
#include "LensContext.h"
#include "ParameterSet.h"
#include <memory>
#include <map>

class SrcRefSumRsqrdInvWeightModifier : public CG_SrcRefSumRsqrdInvWeightModifierBase
{
   public:
      void userInitialize(LensContext* CG_c, Functor*& f, int& maxDim, bool& setDistance);
      std::unique_ptr<ParameterSet> userExecute(LensContext* CG_c);
      SrcRefSumRsqrdInvWeightModifier();
      virtual ~SrcRefSumRsqrdInvWeightModifier();
      virtual void duplicate(std::unique_ptr<SrcRefSumRsqrdInvWeightModifier>&& dup) const;
      virtual void duplicate(std::unique_ptr<Functor>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_SrcRefSumRsqrdInvWeightModifierBase>&& dup) const;

      bool _setDistance;
      std::map<float, float> _radiusMap;
      int _maxDistance;
};

#endif
