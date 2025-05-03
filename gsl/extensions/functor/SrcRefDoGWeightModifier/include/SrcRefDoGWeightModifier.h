// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef SrcRefDoGWeightModifier_H
#define SrcRefDoGWeightModifier_H
#include "Mgs.h"

#include "CG_SrcRefDoGWeightModifierBase.h"
#include "GslContext.h"
#include "ParameterSet.h"
#include <memory>

class SrcRefDoGWeightModifier : public CG_SrcRefDoGWeightModifierBase
{
   public:
      void userInitialize(GslContext* CG_c, Functor*& f, float& sigma1, float& max1, float& sigma2, float& max2, int& wrapDistance);
      std::unique_ptr<ParameterSet> userExecute(GslContext* CG_c);
      SrcRefDoGWeightModifier();
      virtual ~SrcRefDoGWeightModifier();
      virtual void duplicate(std::unique_ptr<SrcRefDoGWeightModifier>&& dup) const;
      virtual void duplicate(std::unique_ptr<Functor>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_SrcRefDoGWeightModifierBase>&& dup) const;

      float _sigma1;
      float _max1;
      float _sigma2;
      float _max2;
      int _wrapDistance;
};

#endif
