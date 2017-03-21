#ifndef GatedThalamicUnitCompCategory_H
#define GatedThalamicUnitCompCategory_H

#include "Lens.h"
#include "CG_GatedThalamicUnitCompCategory.h"

class NDPairList;

class GatedThalamicUnitCompCategory : public CG_GatedThalamicUnitCompCategory
{
   public:
      GatedThalamicUnitCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void initializeShared(RNG& rng);
      void outputWeightsShared(RNG& rng);
      void inputWeightsShared(RNG& rng);
};

#endif
