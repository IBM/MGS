// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef DstRefDistanceModifier_H
#define DstRefDistanceModifier_H
#include "Mgs.h"

#include "CG_DstRefDistanceModifierBase.h"
#include "GslContext.h"
#include "ParameterSet.h"
#include <memory>

class DstRefDistanceModifier : public CG_DstRefDistanceModifierBase
{
   public:
      void userInitialize(GslContext* CG_c, Functor*& f);
      std::unique_ptr<ParameterSet> userExecute(GslContext* CG_c);
      DstRefDistanceModifier();
      virtual ~DstRefDistanceModifier();
      virtual void duplicate(std::unique_ptr<DstRefDistanceModifier>&& dup) const;
      virtual void duplicate(std::unique_ptr<Functor>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_DstRefDistanceModifierBase>&& dup) const;
};

#endif
