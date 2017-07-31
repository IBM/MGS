#ifndef AtasoyNFUnitCompCategory_H
#define AtasoyNFUnitCompCategory_H

#include "Lens.h"
#include "CG_AtasoyNFUnitCompCategory.h"

class NDPairList;

class AtasoyNFUnitCompCategory : public CG_AtasoyNFUnitCompCategory
{
   public:
      AtasoyNFUnitCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void initializeShared(RNG& rng);
};

#endif
