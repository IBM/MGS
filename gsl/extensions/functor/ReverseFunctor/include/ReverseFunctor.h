// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef ReverseFunctor_H
#define ReverseFunctor_H
#include "Mgs.h"

#include "CG_ReverseFunctorBase.h"
#include "LensContext.h"
#include <memory>

class ReverseFunctor : public CG_ReverseFunctorBase
{
   public:
      void userInitialize(LensContext* CG_c, Functor*& f);
      void userExecute(LensContext* CG_c);
      ReverseFunctor();
      virtual ~ReverseFunctor();
      virtual void duplicate(std::unique_ptr<ReverseFunctor>&& dup) const;
      virtual void duplicate(std::unique_ptr<Functor>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_ReverseFunctorBase>&& dup) const;
};
#endif
