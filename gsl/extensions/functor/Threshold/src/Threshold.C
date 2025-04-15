// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Lens.h"
#include "Threshold.h"
#include "CG_ThresholdBase.h"
#include "LensContext.h"
#include <memory>

void Threshold::userInitialize(LensContext* CG_c, Functor*& f, double& threshold) 
{
}

bool Threshold::userExecute(LensContext* CG_c) 
{
  std::vector<DataItem*> nullArgs;
  std::unique_ptr<DataItem> rval_ap;
  init.f->execute(CG_c, nullArgs, rval_ap);
  NumericDataItem *ndi = 
    dynamic_cast<NumericDataItem*>(rval_ap.get());
  if (ndi==0) {
    throw SyntaxErrorException("Threshold, second argument: functor did not return a number");
  }
  return (ndi->getDouble()>init.threshold);
}

Threshold::Threshold() 
   : CG_ThresholdBase()
{
}

Threshold::~Threshold() 
{
}

void Threshold::duplicate(std::unique_ptr<Threshold>&& dup) const
{
   dup.reset(new Threshold(*this));
}

void Threshold::duplicate(std::unique_ptr<Functor>&& dup) const
{
   dup.reset(new Threshold(*this));
}

void Threshold::duplicate(std::unique_ptr<CG_ThresholdBase>&& dup) const
{
   dup.reset(new Threshold(*this));
}

