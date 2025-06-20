// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef DstRefSumRsqrdInvWeightModifier_H
#define DstRefSumRsqrdInvWeightModifier_H
#include "Mgs.h"

#include "CG_DstRefSumRsqrdInvWeightModifierBase.h"
#include "GslContext.h"
#include "ParameterSet.h"
#include <memory>
#include <map>

class DstRefSumRsqrdInvWeightModifier : public CG_DstRefSumRsqrdInvWeightModifierBase
{
   public:
      void userInitialize(GslContext* CG_c, Functor*& f, int& maxDim, bool& setDistance);
      std::unique_ptr<ParameterSet> userExecute(GslContext* CG_c);
      DstRefSumRsqrdInvWeightModifier();
      virtual ~DstRefSumRsqrdInvWeightModifier();
      virtual void duplicate(std::unique_ptr<DstRefSumRsqrdInvWeightModifier>&& dup) const;
      virtual void duplicate(std::unique_ptr<Functor>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_DstRefSumRsqrdInvWeightModifierBase>&& dup) const;

      bool _setDistance;
      std::map<float, float> _radiusMap;
      int _maxDistance;
};

#endif
