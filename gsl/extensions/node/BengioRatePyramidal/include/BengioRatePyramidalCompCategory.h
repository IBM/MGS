#ifndef BengioRatePyramidalCompCategory_H
#define BengioRatePyramidalCompCategory_H

#include "Lens.h"
#include "CG_BengioRatePyramidalCompCategory.h"

class NDPairList;

class BengioRatePyramidalCompCategory : public CG_BengioRatePyramidalCompCategory
{
   public:
      BengioRatePyramidalCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void outputWeightsShared(RNG& rng);
      void initialize(RNG& rng);
};

#endif
