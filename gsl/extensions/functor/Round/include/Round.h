// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef Round_H
#define Round_H

#include "Mgs.h"
#include "CG_RoundBase.h"
#include "LensContext.h"
#include <memory>

class Round : public CG_RoundBase
{
   public:
      void userInitialize(LensContext* CG_c, Functor*& f);
      double userExecute(LensContext* CG_c);
      Round();
      virtual ~Round();
      virtual void duplicate(std::unique_ptr<Round>&& dup) const;
      virtual void duplicate(std::unique_ptr<Functor>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_RoundBase>&& dup) const;
};

#endif
