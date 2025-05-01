// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef RefAngleModifier_H
#define RefAngleModifier_H
#include "Mgs.h"

#include "CG_RefAngleModifierBase.h"
#include "LensContext.h"
#include "ParameterSet.h"
#include <memory>

class RefAngleModifier : public CG_RefAngleModifierBase
{
   public:
      void userInitialize(LensContext* CG_c, int& directionFlag, int& wrapFlag, Functor*& f);
      std::unique_ptr<ParameterSet> userExecute(LensContext* CG_c);
      RefAngleModifier();
      virtual ~RefAngleModifier();
      virtual void duplicate(std::unique_ptr<RefAngleModifier>&& dup) const;
      virtual void duplicate(std::unique_ptr<Functor>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_RefAngleModifierBase>&& dup) const;
};

#endif
