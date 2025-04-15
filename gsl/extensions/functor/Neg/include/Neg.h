// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef Neg_H
#define Neg_H

#include "Lens.h"
#include "CG_NegBase.h"
#include "LensContext.h"
#include <memory>

class Neg : public CG_NegBase
{
   public:
      void userInitialize(LensContext* CG_c, Functor*& f);
      double userExecute(LensContext* CG_c);
      Neg();
      virtual ~Neg();
      virtual void duplicate(std::unique_ptr<Neg>&& dup) const;
      virtual void duplicate(std::unique_ptr<Functor>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_NegBase>&& dup) const;
};

#endif
