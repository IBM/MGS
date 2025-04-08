#include "Lens.h"
#include "CombineNVPairs.h"
#include "CG_CombineNVPairsBase.h"
#include "LensContext.h"
#include "NDPairList.h"
#include "NDPairListDataItem.h"
#include <memory>

void CombineNVPairs::userInitialize(LensContext* CG_c, NDPairList*& l, DuplicatePointerArray< Functor >& fl) 
{
}

std::unique_ptr<NDPairList> CombineNVPairs::userExecute(LensContext* CG_c) 
{
  DuplicatePointerArray< Functor >::iterator iter = init.fl.begin(), end = init.fl.end();
  std::unique_ptr<NDPairList> rval;
  init.l->duplicate(std::move(rval));

  for (; iter!=end; ++iter) {
    std::vector<DataItem*> nullArgs;
    std::unique_ptr<DataItem> rval_ap;
    (*iter)->execute(CG_c, nullArgs, rval_ap);
    NDPairListDataItem *ndi = 
      dynamic_cast<NDPairListDataItem*>(rval_ap.get());
    if (ndi==0) {
      throw SyntaxErrorException("CombineNVPairs, second argument: functor did not return a number");
    }
    rval->splice(rval->end(), *(ndi->getNDPairList()));
  }
  return rval;
}

CombineNVPairs::CombineNVPairs() 
   : CG_CombineNVPairsBase()
{
}

CombineNVPairs::~CombineNVPairs() 
{
}

void CombineNVPairs::duplicate(std::unique_ptr<CombineNVPairs>&& dup) const
{
   dup.reset(new CombineNVPairs(*this));
}

void CombineNVPairs::duplicate(std::unique_ptr<Functor>&& dup) const
{
   dup.reset(new CombineNVPairs(*this));
}

void CombineNVPairs::duplicate(std::unique_ptr<CG_CombineNVPairsBase>&& dup) const
{
   dup.reset(new CombineNVPairs(*this));
}

