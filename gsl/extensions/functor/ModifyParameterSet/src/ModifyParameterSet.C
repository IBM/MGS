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
   psdi->getParameterSet()->duplicate(pset);

   init.f2->execute(CG_c, nullArgs, rval_ap);
   NDPairListDataItem *ndpldi = 
      dynamic_cast<NDPairListDataItem*>(rval_ap.release());
   if (ndpldi==0) {
      throw SyntaxErrorException(
	 "ModifyParameterSet, second argument: functor did not return a NDPairList");
   }
   std::unique_ptr<NDPairList> ndpl_aptr;
   ndpldi->getNDPairList()->duplicate(ndpl_aptr);

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

void ModifyParameterSet::duplicate(std::unique_ptr<ModifyParameterSet>& dup) const
{
   dup.reset(new ModifyParameterSet(*this));
}

void ModifyParameterSet::duplicate(std::unique_ptr<Functor>& dup) const
{
   dup.reset(new ModifyParameterSet(*this));
}

void ModifyParameterSet::duplicate(std::unique_ptr<CG_ModifyParameterSetBase>& dup) const
{
   dup.reset(new ModifyParameterSet(*this));
}

