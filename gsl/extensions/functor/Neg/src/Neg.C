// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Mgs.h"
#include "Neg.h"
#include "CG_NegBase.h"
#include "GslContext.h"
#include <memory>

void Neg::userInitialize(GslContext* CG_c, Functor*& f) 
{
}

double Neg::userExecute(GslContext* CG_c) 
{
  std::vector<DataItem*> nullArgs;
  std::unique_ptr<DataItem> rval_ap;
  init.f->execute(CG_c, nullArgs, rval_ap);
  NumericDataItem *ndi = 
    dynamic_cast<NumericDataItem*>(rval_ap.get());
  if (ndi==0) {
    throw SyntaxErrorException("Neg, second argument: functor did not return a number");
  }
  return -ndi->getDouble();
}

Neg::Neg() 
   : CG_NegBase()
{
}

Neg::~Neg() 
{
}

void Neg::duplicate(std::unique_ptr<Neg>&& dup) const
{
   dup.reset(new Neg(*this));
}

void Neg::duplicate(std::unique_ptr<Functor>&& dup) const
{
   dup.reset(new Neg(*this));
}

void Neg::duplicate(std::unique_ptr<CG_NegBase>&& dup) const
{
   dup.reset(new Neg(*this));
}

