#ifndef GatedThalamoCorticalUnitCompCategory_H
#define GatedThalamoCorticalUnitCompCategory_H

#include "Lens.h"
#include "CG_GatedThalamoCorticalUnitCompCategory.h"

class NDPairList;

class GatedThalamoCorticalUnitCompCategory : public CG_GatedThalamoCorticalUnitCompCategory
{
   public:
      GatedThalamoCorticalUnitCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void initializeShared(RNG& rng);
      void outputWeightsShared(RNG& rng);
      void inputWeightsShared(RNG& rng);
      void updateWhitMatrixShared(RNG& rng);
};

#endif
