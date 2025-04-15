// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Lens.h"
#include "Round.h"
#include "CG_RoundBase.h"
#include "LensContext.h"
#include <memory>

void Round::userInitialize(LensContext* CG_c, Functor*& f) 
{
}

double Round::userExecute(LensContext* CG_c) 
{
  std::vector<DataItem*> nullArgs;
  std::unique_ptr<DataItem> rval_ap;
  init.f->execute(CG_c, nullArgs, rval_ap);
  NumericDataItem *ndi = 
    dynamic_cast<NumericDataItem*>(rval_ap.get());
  if (ndi==0) {
    throw SyntaxErrorException("Round, second argument: functor did not return a number");
  }  
  return round(ndi->getDouble());
}

Round::Round() 
   : CG_RoundBase()
{
}

Round::~Round() 
{
}

void Round::duplicate(std::unique_ptr<Round>&& dup) const
{
   dup.reset(new Round(*this));
}

void Round::duplicate(std::unique_ptr<Functor>&& dup) const
{
   dup.reset(new Round(*this));
}

void Round::duplicate(std::unique_ptr<CG_RoundBase>&& dup) const
{
   dup.reset(new Round(*this));
}

