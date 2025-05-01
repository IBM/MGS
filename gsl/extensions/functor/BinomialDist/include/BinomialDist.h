// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef BinomialDist_H
#define BinomialDist_H

#include "Mgs.h"
#include "CG_BinomialDistBase.h"
#include "LensContext.h"
#include "rndm.h"
#include <memory>

class BinomialDist : public CG_BinomialDistBase
{
   public:
      void userInitialize(LensContext* CG_c, double& prob, double& n1, double& n2, unsigned& seed);
      double userExecute(LensContext* CG_c);
      BinomialDist();
      BinomialDist(const BinomialDist&);
      virtual ~BinomialDist();
      virtual void duplicate(std::unique_ptr<BinomialDist>&& dup) const;
      virtual void duplicate(std::unique_ptr<Functor>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_BinomialDistBase>&& dup) const;

      //   private:
      //      RNG _rng;
};

#endif
