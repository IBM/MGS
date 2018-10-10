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

void Round::duplicate(std::unique_ptr<Round>& dup) const
{
   dup.reset(new Round(*this));
}

void Round::duplicate(std::unique_ptr<Functor>& dup) const
{
   dup.reset(new Round(*this));
}

void Round::duplicate(std::unique_ptr<CG_RoundBase>& dup) const
{
   dup.reset(new Round(*this));
}

