// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Mgs.h"
#include "NameReturnValue.h"
#include "CG_NameReturnValueBase.h"
#include "LensContext.h"
#include "NDPairList.h"
#include "NumericDataItem.h"
#include <memory>

void NameReturnValue::userInitialize(LensContext* CG_c, CustomString& s, Functor*& f) 
{
}

std::unique_ptr<NDPairList> NameReturnValue::userExecute(LensContext* CG_c) 
{
  std::vector<DataItem*> nullArgs;
  std::unique_ptr<DataItem> rval_ap;
  init.f->execute(CG_c, nullArgs, rval_ap);
  NumericDataItem *ndi = 
    dynamic_cast<NumericDataItem*>(rval_ap.get());
  if (ndi==0) {
    throw SyntaxErrorException("NameReturnValue, second argument: functor did not return a number");
  }

  std::string s(init.s.c_str());
  NDPair* ndp = new NDPair(s, rval_ap);
  std::unique_ptr<NDPairList> rval(new NDPairList());  
  rval->push_back(ndp);
  return rval;
}

NameReturnValue::NameReturnValue() 
   : CG_NameReturnValueBase()
{
}

NameReturnValue::~NameReturnValue() 
{
}

void NameReturnValue::duplicate(std::unique_ptr<NameReturnValue>&& dup) const
{
   dup.reset(new NameReturnValue(*this));
}

void NameReturnValue::duplicate(std::unique_ptr<Functor>&& dup) const
{
   dup.reset(new NameReturnValue(*this));
}

void NameReturnValue::duplicate(std::unique_ptr<CG_NameReturnValueBase>&& dup) const
{
   dup.reset(new NameReturnValue(*this));
}

