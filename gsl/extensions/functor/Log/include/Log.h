// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef Log_H
#define Log_H

#include "Lens.h"
#include "CG_LogBase.h"
#include "LensContext.h"
#include <memory>

class Log : public CG_LogBase
{
   public:
      void userInitialize(LensContext* CG_c, Functor*& f);
      double userExecute(LensContext* CG_c);
      Log();
      virtual ~Log();
      virtual void duplicate(std::unique_ptr<Log>&& dup) const;
      virtual void duplicate(std::unique_ptr<Functor>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_LogBase>&& dup) const;
};

#endif
