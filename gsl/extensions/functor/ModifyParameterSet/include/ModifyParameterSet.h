// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef ModifyParameterSet_H
#define ModifyParameterSet_H

#include "Mgs.h"
#include "CG_ModifyParameterSetBase.h"
#include "LensContext.h"
#include "ParameterSet.h"
#include <memory>

class ModifyParameterSet : public CG_ModifyParameterSetBase
{
   public:
      void userInitialize(LensContext* CG_c, Functor*& f1, Functor*& f2);
      std::unique_ptr<ParameterSet> userExecute(LensContext* CG_c);
      ModifyParameterSet();
      virtual ~ModifyParameterSet();
      virtual void duplicate(std::unique_ptr<ModifyParameterSet>&& dup) const;
      virtual void duplicate(std::unique_ptr<Functor>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_ModifyParameterSetBase>&& dup) const;
};

#endif
