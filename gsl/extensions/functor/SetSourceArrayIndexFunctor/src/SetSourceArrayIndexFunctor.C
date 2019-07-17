#include "Lens.h"
#include "SetSourceArrayIndexFunctor.h"
#include "CG_SetSourceArrayIndexFunctorBase.h"
#include "LensContext.h"
#include "ConnectionContext.h"
#include "ParameterSetDataItem.h"
#include "NodeDescriptor.h"
#include "ParameterSet.h"
#include "NDPairList.h"
#include "NDPair.h"
#include "UnsignedIntDataItem.h"
#include "Simulation.h"

#include <memory>

void SetSourceArrayIndexFunctor::userInitialize(LensContext* CG_c, Functor*& destinationInAttr) 
{
  destinationInAttr->duplicate(_destinationInAttr);
}

std::auto_ptr<ParameterSet> SetSourceArrayIndexFunctor::userExecute(LensContext* CG_c) 
{
  ConnectionContext* cc = CG_c->connectionContext;
  if (cc->restart) _indexMap.clear();

  std::vector<DataItem*> nullArgs;
  std::auto_ptr<DataItem> inAttrRVal;
  std::auto_ptr<ParameterSet> rval;
  
  _destinationInAttr->execute(CG_c, nullArgs, inAttrRVal);

  ParameterSetDataItem *psdi = dynamic_cast<ParameterSetDataItem*>(inAttrRVal.get());
  if (psdi==0) {
    throw SyntaxErrorException("BidirectConnectNodeSets: InAttrPSet functor did not return a Parameter Set!");
  }
  ParameterSet* pset = psdi->getParameterSet();
  
  unsigned thisIndex=0;
  if (_indexMap.find(cc->sourceNode) == _indexMap.end())
    _indexMap[cc->sourceNode]=0;
  else thisIndex = ++_indexMap[cc->sourceNode];
  
  NDPairList paramsLocal;
  UnsignedIntDataItem* paramDI = new UnsignedIntDataItem(thisIndex);
  std::auto_ptr<DataItem> paramDI_ap(paramDI);
  NDPair* ndp = new NDPair("index", paramDI_ap);
  paramsLocal.push_back(ndp);

  pset->set(paramsLocal);
  pset->duplicate(rval);
  return rval;
}

SetSourceArrayIndexFunctor::SetSourceArrayIndexFunctor() 
   : CG_SetSourceArrayIndexFunctorBase()
{
}

SetSourceArrayIndexFunctor::SetSourceArrayIndexFunctor(SetSourceArrayIndexFunctor const& f)
    : CG_SetSourceArrayIndexFunctorBase(f), _indexMap(f._indexMap)
{
  f._destinationInAttr->duplicate(_destinationInAttr);
}

SetSourceArrayIndexFunctor::~SetSourceArrayIndexFunctor() 
{
}

void SetSourceArrayIndexFunctor::duplicate(std::auto_ptr<SetSourceArrayIndexFunctor>& dup) const
{
   dup.reset(new SetSourceArrayIndexFunctor(*this));
}

void SetSourceArrayIndexFunctor::duplicate(std::auto_ptr<Functor>& dup) const
{
   dup.reset(new SetSourceArrayIndexFunctor(*this));
}

void SetSourceArrayIndexFunctor::duplicate(std::auto_ptr<CG_SetSourceArrayIndexFunctorBase>& dup) const
{
   dup.reset(new SetSourceArrayIndexFunctor(*this));
}

