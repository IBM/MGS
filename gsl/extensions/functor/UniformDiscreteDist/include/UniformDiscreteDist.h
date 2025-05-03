// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef UniformDiscreteDist_H
#define UniformDiscreteDist_H

#include "Mgs.h"
#include "CG_UniformDiscreteDistBase.h"
#include "GslContext.h"
#include <memory>

class UniformDiscreteDist : public CG_UniformDiscreteDistBase
{
   public:
      void userInitialize(GslContext* CG_c, double& n1, double& n2);
      int userExecute(GslContext* CG_c);
      UniformDiscreteDist();
      virtual ~UniformDiscreteDist();
      virtual void duplicate(std::unique_ptr<UniformDiscreteDist>&& dup) const;
      virtual void duplicate(std::unique_ptr<Functor>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_UniformDiscreteDistBase>&& dup) const;
};

#endif
