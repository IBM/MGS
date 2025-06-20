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

#include "Mgs.h"
#include "CG_NegBase.h"
#include "GslContext.h"
#include <memory>

class Neg : public CG_NegBase
{
   public:
      void userInitialize(GslContext* CG_c, Functor*& f);
      double userExecute(GslContext* CG_c);
      Neg();
      virtual ~Neg();
      virtual void duplicate(std::unique_ptr<Neg>&& dup) const;
      virtual void duplicate(std::unique_ptr<Functor>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_NegBase>&& dup) const;
};

#endif
