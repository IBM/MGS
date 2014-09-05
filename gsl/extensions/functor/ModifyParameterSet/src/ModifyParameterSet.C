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

std::auto_ptr<ParameterSet> ModifyParameterSet::userExecute(LensContext* CG_c) 
{
   std::vector<DataItem*> nullArgs;
   std::auto_ptr<DataItem> rval_ap;

   init.f1->execute(CG_c, nullArgs, rval_ap);
   ParameterSetDataItem *psdi = 
      dynamic_cast<ParameterSetDataItem*>(rval_ap.release());
   if (psdi==0) {
      throw SyntaxErrorException(
	 "ModifyParameterSet, first argument: functor did not return a Parameter Set");
   }
   std::auto_ptr<ParameterSet> pset;
   psdi->getParameterSet()->duplicate(pset);

   init.f2->execute(CG_c, nullArgs, rval_ap);
   NDPairListDataItem *ndpldi = 
      dynamic_cast<NDPairListDataItem*>(rval_ap.release());
   if (ndpldi==0) {
      throw SyntaxErrorException(
	 "ModifyParameterSet, second argument: functor did not return a NDPairList");
   }
   std::auto_ptr<NDPairList> ndpl_aptr;
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

void ModifyParameterSet::duplicate(std::auto_ptr<ModifyParameterSet>& dup) const
{
   dup.reset(new ModifyParameterSet(*this));
}

void ModifyParameterSet::duplicate(std::auto_ptr<Functor>& dup) const
{
   dup.reset(new ModifyParameterSet(*this));
}

void ModifyParameterSet::duplicate(std::auto_ptr<CG_ModifyParameterSetBase>& dup) const
{
   dup.reset(new ModifyParameterSet(*this));
}

