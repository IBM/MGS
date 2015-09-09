#ifndef LifeNodeCompCategory_H
#define LifeNodeCompCategory_H

#include "Lens.h"
#include "CG_LifeNodeCompCategory.h"

class NDPairList;

class LifeNodeCompCategory : public CG_LifeNodeCompCategory
{
   public:
      LifeNodeCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
};

#endif
