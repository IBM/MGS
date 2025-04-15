// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef SrcRefDistanceModifier_H
#define SrcRefDistanceModifier_H
#include "Lens.h"

#include "CG_SrcRefDistanceModifierBase.h"
#include "LensContext.h"
#include "ParameterSet.h"
#include <memory>

class SrcRefDistanceModifier : public CG_SrcRefDistanceModifierBase
{
   public:
      void userInitialize(LensContext* CG_c, Functor*& f);
      std::unique_ptr<ParameterSet> userExecute(LensContext* CG_c);
      SrcRefDistanceModifier();
      virtual ~SrcRefDistanceModifier();
      virtual void duplicate(std::unique_ptr<SrcRefDistanceModifier>&& dup) const;
      virtual void duplicate(std::unique_ptr<Functor>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_SrcRefDistanceModifierBase>&& dup) const;
};

#endif
