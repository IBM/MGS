#include "Lens.h"
#include "FloatArrayMaker.h"
#include "CG_FloatArrayMakerBase.h"
#include "LensContext.h"
#include "ShallowArray.h"
#include <memory>

void FloatArrayMaker::userInitialize(LensContext* CG_c, Functor*& f, int& size) 
{
}

ShallowArray< float > FloatArrayMaker::userExecute(LensContext* CG_c) 
{
  ShallowArray<float> this_rval;
  this_rval.increaseSizeTo(init.size);
  for (int n=0; n<init.size; ++n) {
    std::vector<DataItem*> nullArgs;
    std::auto_ptr<DataItem> rval_ap;
    init.f->execute(CG_c, nullArgs, rval_ap);
    NumericDataItem *ndi = 
      dynamic_cast<NumericDataItem*>(rval_ap.get());
    float value = 0;
    if (ndi==0) {
      throw 
	SyntaxErrorException("FloatArrayMaker, second argument: functor did not return a number");
    }
    else value = ndi->getFloat();
    this_rval[n] = value;
  }
  return this_rval;
}

FloatArrayMaker::FloatArrayMaker() 
   : CG_FloatArrayMakerBase()
{
}

FloatArrayMaker::~FloatArrayMaker() 
{
}

void FloatArrayMaker::duplicate(std::auto_ptr<FloatArrayMaker>& dup) const
{
   dup.reset(new FloatArrayMaker(*this));
}

void FloatArrayMaker::duplicate(std::auto_ptr<Functor>& dup) const
{
   dup.reset(new FloatArrayMaker(*this));
}

void FloatArrayMaker::duplicate(std::auto_ptr<CG_FloatArrayMakerBase>& dup) const
{
   dup.reset(new FloatArrayMaker(*this));
}

