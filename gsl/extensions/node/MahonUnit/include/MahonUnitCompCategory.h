#ifndef MahonUnitCompCategory_H
#define MahonUnitCompCategory_H

#include "Lens.h"
#include "CG_MahonUnitCompCategory.h"

class NDPairList;

class MahonUnitCompCategory : public CG_MahonUnitCompCategory
{
   public:
      MahonUnitCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
};

#endif
