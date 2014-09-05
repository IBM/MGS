#include "Lens.h"
#include "BinomialDist.h"
#include "CG_BinomialDistBase.h"
#include "LensContext.h"
#include <memory>

void BinomialDist::userInitialize(LensContext* CG_c, double& probOfN1, double& n1, double& n2, unsigned& seed) 
{
  _rng.reSeedShared(seed);
}

double BinomialDist::userExecute(LensContext* CG_c) 
{
  double rval=init.n2;
  if (drandom(0, 1.0, _rng)<init.probOfN1) {
    rval=init.n1;
  }
  return rval;
}

BinomialDist::BinomialDist() 
  : CG_BinomialDistBase()
{
}

BinomialDist::BinomialDist(const BinomialDist& rv)
  : CG_BinomialDistBase(), _rng(rv._rng)
{
}

BinomialDist::~BinomialDist() 
{
}

void BinomialDist::duplicate(std::auto_ptr<BinomialDist>& dup) const
{
   dup.reset(new BinomialDist(*this));
}

void BinomialDist::duplicate(std::auto_ptr<Functor>& dup) const
{
   dup.reset(new BinomialDist(*this));
}

void BinomialDist::duplicate(std::auto_ptr<CG_BinomialDistBase>& dup) const
{
   dup.reset(new BinomialDist(*this));
}

