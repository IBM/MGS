#ifndef SwitchInputCompCategory_H
#define SwitchInputCompCategory_H

#include "Lens.h"
#include "CG_SwitchInputCompCategory.h"

class NDPairList;

class SwitchInputCompCategory : public CG_SwitchInputCompCategory
{
   public:
      SwitchInputCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void initializeShared(RNG& rng);
      void updateInputState(RNG& rng);
};

#endif
