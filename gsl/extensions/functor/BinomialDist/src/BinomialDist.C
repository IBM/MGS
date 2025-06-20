// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Mgs.h"
#include "BinomialDist.h"
#include "CG_BinomialDistBase.h"
#include "GslContext.h"
#include "Simulation.h"
#include <memory>

void BinomialDist::userInitialize(GslContext* CG_c, double& probOfN1, double& n1, double& n2, unsigned& seed) 
{
  //  _rng.reSeedShared(seed);
}

double BinomialDist::userExecute(GslContext* CG_c) 
{
  double rval=init.n2;
  if (drandom(0, 1.0, CG_c->sim->getSharedFunctorRandomSeedGenerator())<init.probOfN1) {
    rval=init.n1;
  }
  return rval;
}

BinomialDist::BinomialDist() 
  : CG_BinomialDistBase()
{
}

BinomialDist::BinomialDist(const BinomialDist& rv)
//  : CG_BinomialDistBase(), _rng(rv._rng)
{
}

BinomialDist::~BinomialDist() 
{
}

void BinomialDist::duplicate(std::unique_ptr<BinomialDist>&& dup) const
{
   dup.reset(new BinomialDist(*this));
}

void BinomialDist::duplicate(std::unique_ptr<Functor>&& dup) const
{
   dup.reset(new BinomialDist(*this));
}

void BinomialDist::duplicate(std::unique_ptr<CG_BinomialDistBase>&& dup) const
{
   dup.reset(new BinomialDist(*this));
}

