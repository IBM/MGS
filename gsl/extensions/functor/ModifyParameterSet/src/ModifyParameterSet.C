// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Lens.h"
#include "ModifyParameterSet.h"
#include "CG_ModifyParameterSetBase.h"
#include "LensContext.h"
#include "ParameterSet.h"
#include "NDPairListDataItem.h"
#include <memory>

void ModifyParameterSet::userInitialize(LensContext* CG_c, Functor*& f1, Functor*& f2)
{
}

std::unique_ptr<ParameterSet> ModifyParameterSet::userExecute(LensContext* CG_c) 
{
   std::vector<DataItem*> nullArgs;
   std::unique_ptr<DataItem> rval_ap;

   init.f1->execute(CG_c, nullArgs, rval_ap);
   ParameterSetDataItem *psdi = 
      dynamic_cast<ParameterSetDataItem*>(rval_ap.release());
   if (psdi==0) {
      throw SyntaxErrorException(
	 "ModifyParameterSet, first argument: functor did not return a Parameter Set");
   }
   std::unique_ptr<ParameterSet> pset;
   psdi->getParameterSet()->duplicate(std::move(pset));

   init.f2->execute(CG_c, nullArgs, rval_ap);
   NDPairListDataItem *ndpldi = 
      dynamic_cast<NDPairListDataItem*>(rval_ap.release());
   if (ndpldi==0) {
      throw SyntaxErrorException(
	 "ModifyParameterSet, second argument: functor did not return a NDPairList");
   }
   std::unique_ptr<NDPairList> ndpl_aptr;
   ndpldi->getNDPairList()->duplicate(std::move(ndpl_aptr));

   pset->set(*(ndpl_aptr.release()));
   return pset;
}

ModifyParameterSet::ModifyParameterSet() 
   : CG_ModifyParameterSetBase()
{
}

ModifyParameterSet::~ModifyParameterSet() 
{
}

void ModifyParameterSet::duplicate(std::unique_ptr<ModifyParameterSet>&& dup) const
{
   dup.reset(new ModifyParameterSet(*this));
}

void ModifyParameterSet::duplicate(std::unique_ptr<Functor>&& dup) const
{
   dup.reset(new ModifyParameterSet(*this));
}

void ModifyParameterSet::duplicate(std::unique_ptr<CG_ModifyParameterSetBase>&& dup) const
{
   dup.reset(new ModifyParameterSet(*this));
}

