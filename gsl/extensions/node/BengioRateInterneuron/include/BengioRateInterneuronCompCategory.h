#ifndef BengioRateInterneuronCompCategory_H
#define BengioRateInterneuronCompCategory_H

#include "Lens.h"
#include "CG_BengioRateInterneuronCompCategory.h"

class NDPairList;

class BengioRateInterneuronCompCategory : public CG_BengioRateInterneuronCompCategory
{
   public:
      BengioRateInterneuronCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void outputWeightsShared(RNG& rng);
      void initialize(RNG& rng);
};

#endif
