// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef RefDistanceModifier_H
#define RefDistanceModifier_H
#include "Mgs.h"

#include "CG_RefDistanceModifierBase.h"
#include "LensContext.h"
#include "ParameterSet.h"
#include <memory>

class RefDistanceModifier : public CG_RefDistanceModifierBase
{
   public:
      void userInitialize(LensContext* CG_c, int& directionFlag, int& WrapFlag, Functor*& f);
      std::unique_ptr<ParameterSet> userExecute(LensContext* CG_c);
      RefDistanceModifier();
      virtual ~RefDistanceModifier();
      virtual void duplicate(std::unique_ptr<RefDistanceModifier>&& dup) const;
      virtual void duplicate(std::unique_ptr<Functor>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_RefDistanceModifierBase>&& dup) const;
};

#endif
