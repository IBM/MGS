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

   private:
      RNG _rng;
};

#endif
