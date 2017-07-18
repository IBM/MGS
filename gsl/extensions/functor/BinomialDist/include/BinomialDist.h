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

#ifndef BinomialDist_H
#define BinomialDist_H

#include "Lens.h"
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
      virtual void duplicate(std::auto_ptr<BinomialDist>& dup) const;
      virtual void duplicate(std::auto_ptr<Functor>& dup) const;
      virtual void duplicate(std::auto_ptr<CG_BinomialDistBase>& dup) const;

      //   private:
      //      RNG _rng;
};

#endif
