// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef FloatArrayMaker_H
#define FloatArrayMaker_H

#include "Mgs.h"
#include "CG_FloatArrayMakerBase.h"
#include "GslContext.h"
#include "ShallowArray.h"
#include <memory>

class FloatArrayMaker : public CG_FloatArrayMakerBase
{
   public:
      void userInitialize(GslContext* CG_c, Functor*& f, int& size);
      ShallowArray< float > userExecute(GslContext* CG_c);
      FloatArrayMaker();
      virtual ~FloatArrayMaker();
      virtual void duplicate(std::unique_ptr<FloatArrayMaker>&& dup) const;
      virtual void duplicate(std::unique_ptr<Functor>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_FloatArrayMakerBase>&& dup) const;
};

#endif
