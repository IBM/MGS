#ifndef ToneUnitCompCategory_H
#define ToneUnitCompCategory_H

#include "Lens.h"
#include "CG_ToneUnitCompCategory.h"

class NDPairList;

class ToneUnitCompCategory : public CG_ToneUnitCompCategory
{
   public:
      ToneUnitCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void initializeShared(RNG& rng);
};

#endif
