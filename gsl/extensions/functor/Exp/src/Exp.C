// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Mgs.h"
#include "Exp.h"
#include "CG_ExpBase.h"
#include "LensContext.h"
#include <memory>

void Exp::userInitialize(LensContext* CG_c, Functor*& f) 
{
}

double Exp::userExecute(LensContext* CG_c) 
{
  std::vector<DataItem*> nullArgs;
  std::unique_ptr<DataItem> rval_ap;
  init.f->execute(CG_c, nullArgs, rval_ap);
  NumericDataItem *ndi = 
    dynamic_cast<NumericDataItem*>(rval_ap.get());
  if (ndi==0) {
    throw SyntaxErrorException("Log, second argument: functor did not return a number");
  }
  return exp(ndi->getDouble());
}

Exp::Exp() 
   : CG_ExpBase()
{
}

Exp::~Exp() 
{
}

void Exp::duplicate(std::unique_ptr<Exp>&& dup) const
{
   dup.reset(new Exp(*this));
}

void Exp::duplicate(std::unique_ptr<Functor>&& dup) const
{
   dup.reset(new Exp(*this));
}

void Exp::duplicate(std::unique_ptr<CG_ExpBase>&& dup) const
{
   dup.reset(new Exp(*this));
}

