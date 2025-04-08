// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "Lens.h"
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

