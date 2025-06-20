// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Mgs.h"
#include "Log.h"
#include "CG_LogBase.h"
#include "GslContext.h"
#include <memory>
#include <math.h>

void Log::userInitialize(GslContext* CG_c, Functor*& f) 
{
}

double Log::userExecute(GslContext* CG_c) 
{
  std::vector<DataItem*> nullArgs;
  std::unique_ptr<DataItem> rval_ap;
  init.f->execute(CG_c, nullArgs, rval_ap);
  NumericDataItem *ndi = 
    dynamic_cast<NumericDataItem*>(rval_ap.get());
  if (ndi==0) {
    throw SyntaxErrorException("Log, second argument: functor did not return a number");
  }
  return log(ndi->getDouble());
}

Log::Log() 
   : CG_LogBase()
{
}

Log::~Log() 
{
}

void Log::duplicate(std::unique_ptr<Log>&& dup) const
{
   dup.reset(new Log(*this));
}

void Log::duplicate(std::unique_ptr<Functor>&& dup) const
{
   dup.reset(new Log(*this));
}

void Log::duplicate(std::unique_ptr<CG_LogBase>&& dup) const
{
   dup.reset(new Log(*this));
}

