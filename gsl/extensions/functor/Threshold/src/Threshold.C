// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

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
  std::auto_ptr<DataItem> rval_ap;
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

void Threshold::duplicate(std::auto_ptr<Threshold>& dup) const
{
   dup.reset(new Threshold(*this));
}

void Threshold::duplicate(std::auto_ptr<Functor>& dup) const
{
   dup.reset(new Threshold(*this));
}

void Threshold::duplicate(std::auto_ptr<CG_ThresholdBase>& dup) const
{
   dup.reset(new Threshold(*this));
}

