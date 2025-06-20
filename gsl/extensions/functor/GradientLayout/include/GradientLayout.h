// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef GradientLayout_H
#define GradientLayout_H

#include "Mgs.h"
#include "CG_GradientLayoutBase.h"
#include "GslContext.h"
#include "ShallowArray.h"
#include <memory>

class GradientLayout : public CG_GradientLayoutBase
{
   public:
      void userInitialize(GslContext* CG_c, int& total, double& slope, ShallowArray< int >& origin, int& originDensity, ShallowArray< bool >& gradientDimensions);
      ShallowArray< int > userExecute(GslContext* CG_c);
      GradientLayout();
      virtual ~GradientLayout();
      virtual void duplicate(std::unique_ptr<GradientLayout>&& dup) const;
      virtual void duplicate(std::unique_ptr<Functor>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_GradientLayoutBase>&& dup) const;
};

#endif
