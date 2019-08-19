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
#include "Log.h"
#include "CG_LogBase.h"
#include "LensContext.h"
#include <memory>
#include <math.h>

void Log::userInitialize(LensContext* CG_c, Functor*& f) 
{
}

double Log::userExecute(LensContext* CG_c) 
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

void Log::duplicate(std::unique_ptr<Log>& dup) const
{
   dup.reset(new Log(*this));
}

void Log::duplicate(std::unique_ptr<Functor>& dup) const
{
   dup.reset(new Log(*this));
}

void Log::duplicate(std::unique_ptr<CG_LogBase>& dup) const
{
   dup.reset(new Log(*this));
}

