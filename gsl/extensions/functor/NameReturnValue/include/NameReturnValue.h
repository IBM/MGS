// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef NameReturnValue_H
#define NameReturnValue_H

#include "Lens.h"
#include "CG_NameReturnValueBase.h"
#include "LensContext.h"
#include "NDPairList.h"
#include <memory>

class NameReturnValue : public CG_NameReturnValueBase
{
   public:
      void userInitialize(LensContext* CG_c, CustomString& s, Functor*& f);
      std::unique_ptr<NDPairList> userExecute(LensContext* CG_c);
      NameReturnValue();
      virtual ~NameReturnValue();
      virtual void duplicate(std::unique_ptr<NameReturnValue>&& dup) const;
      virtual void duplicate(std::unique_ptr<Functor>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_NameReturnValueBase>&& dup) const;
};

#endif
