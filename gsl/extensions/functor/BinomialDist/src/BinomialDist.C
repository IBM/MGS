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
#include "BinomialDist.h"
#include "CG_BinomialDistBase.h"
#include "LensContext.h"
#include "Simulation.h"
#include <memory>

void BinomialDist::userInitialize(LensContext* CG_c, double& probOfN1, double& n1, double& n2, unsigned& seed) 
{
  //  _rng.reSeedShared(seed);
}

double BinomialDist::userExecute(LensContext* CG_c) 
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

