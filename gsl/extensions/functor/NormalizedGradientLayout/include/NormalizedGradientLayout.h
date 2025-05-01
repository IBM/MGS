// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef NormalizedGradientLayout_H
#define NormalizedGradientLayout_H

#include "Mgs.h"
#include "CG_NormalizedGradientLayoutBase.h"
#include "LensContext.h"
#include "ShallowArray.h"
#include <memory>

class NormalizedGradientLayout : public CG_NormalizedGradientLayoutBase
{
   public:
      void userInitialize(LensContext* CG_c, int& total, double& slope, ShallowArray< int >& origin, ShallowArray< bool >& gradientDimensions);
      ShallowArray< int > userExecute(LensContext* CG_c);
      NormalizedGradientLayout();
      virtual ~NormalizedGradientLayout();
      virtual void duplicate(std::unique_ptr<NormalizedGradientLayout>&& dup) const;
      virtual void duplicate(std::unique_ptr<Functor>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_NormalizedGradientLayoutBase>&& dup) const;
};

#endif
