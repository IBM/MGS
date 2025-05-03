// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef RandomDispersalLayout_H
#define RandomDispersalLayout_H

#include "Mgs.h"
#include "CG_RandomDispersalLayoutBase.h"
#include "GslContext.h"
#include "ShallowArray.h"
#include "rndm.h"
#include <memory>

class RandomDispersalLayout : public CG_RandomDispersalLayoutBase
{
   public:
  void userInitialize(GslContext* CG_c, int& total);
      ShallowArray< int > userExecute(GslContext* CG_c);
      RandomDispersalLayout();
      virtual ~RandomDispersalLayout();
      virtual void duplicate(std::unique_ptr<RandomDispersalLayout>&& dup) const;
      virtual void duplicate(std::unique_ptr<Functor>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_RandomDispersalLayoutBase>&& dup) const;
      //   private:
      //      RNG _rng;
};

#endif
