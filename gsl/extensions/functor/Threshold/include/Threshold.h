// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef Threshold_H
#define Threshold_H

#include "Mgs.h"
#include "CG_ThresholdBase.h"
#include "GslContext.h"
#include <memory>

class Threshold : public CG_ThresholdBase
{
   public:
      void userInitialize(GslContext* CG_c, Functor*& f, double& threshold);
      bool userExecute(GslContext* CG_c);
      Threshold();
      virtual ~Threshold();
      virtual void duplicate(std::unique_ptr<Threshold>&& dup) const;
      virtual void duplicate(std::unique_ptr<Functor>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_ThresholdBase>&& dup) const;
};

#endif
