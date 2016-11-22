#include "Lens.h"
#include "Exp.h"
#include "CG_ExpBase.h"
#include "LensContext.h"
#include <memory>

void Exp::userInitialize(LensContext* CG_c, Functor*& f) 
{
}

double Exp::userExecute(LensContext* CG_c) 
{
  std::vector<DataItem*> nullArgs;
  std::auto_ptr<DataItem> rval_ap;
  init.f->execute(CG_c, nullArgs, rval_ap);
  NumericDataItem *ndi = 
    dynamic_cast<NumericDataItem*>(rval_ap.get());
  if (ndi==0) {
    throw SyntaxErrorException("Log, second argument: functor did not return a number");
  }
  return exp(ndi->getDouble());
}

Exp::Exp() 
   : CG_ExpBase()
{
}

Exp::~Exp() 
{
}

void Exp::duplicate(std::auto_ptr<Exp>& dup) const
{
   dup.reset(new Exp(*this));
}

void Exp::duplicate(std::auto_ptr<Functor>& dup) const
{
   dup.reset(new Exp(*this));
}

void Exp::duplicate(std::auto_ptr<CG_ExpBase>& dup) const
{
   dup.reset(new Exp(*this));
}

