// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef SrcRefPeakedWeightModifier_H
#define SrcRefPeakedWeightModifier_H
#include "Mgs.h"

#include "CG_SrcRefPeakedWeightModifierBase.h"
#include "GslContext.h"
#include "ParameterSet.h"
#include <memory>

class SrcRefPeakedWeightModifier : public CG_SrcRefPeakedWeightModifierBase
{
   public:
      void userInitialize(GslContext* CG_c, Functor*& f, float& sigma, float& max, int& wrapDistance);
      std::unique_ptr<ParameterSet> userExecute(GslContext* CG_c);
      SrcRefPeakedWeightModifier();
      virtual ~SrcRefPeakedWeightModifier();
      virtual void duplicate(std::unique_ptr<SrcRefPeakedWeightModifier>&& dup) const;
      virtual void duplicate(std::unique_ptr<Functor>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_SrcRefPeakedWeightModifierBase>&& dup) const;

      float _sigma;
      float _max;
      int _wrapDistance;
};

#endif
