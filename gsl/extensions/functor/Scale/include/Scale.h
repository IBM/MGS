// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef Scale_H
#define Scale_H

#include "Mgs.h"
#include "CG_ScaleBase.h"
#include "LensContext.h"
#include <memory>

class Scale : public CG_ScaleBase
{
   public:
      void userInitialize(LensContext* CG_c, Functor*& f, double& scale);
      double userExecute(LensContext* CG_c);
      Scale();
      virtual ~Scale();
      virtual void duplicate(std::unique_ptr<Scale>&& dup) const;
      virtual void duplicate(std::unique_ptr<Functor>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_ScaleBase>&& dup) const;
};

#endif
