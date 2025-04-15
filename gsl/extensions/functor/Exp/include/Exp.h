// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef Exp_H
#define Exp_H

#include "Lens.h"
#include "CG_ExpBase.h"
#include "LensContext.h"
#include <memory>

class Exp : public CG_ExpBase
{
   public:
      void userInitialize(LensContext* CG_c, Functor*& f);
      double userExecute(LensContext* CG_c);
      Exp();
      virtual ~Exp();
      virtual void duplicate(std::unique_ptr<Exp>&& dup) const;
      virtual void duplicate(std::unique_ptr<Functor>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_ExpBase>&& dup) const;
};

#endif
