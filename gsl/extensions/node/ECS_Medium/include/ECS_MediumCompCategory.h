#ifndef ECS_MediumCompCategory_H
#define ECS_MediumCompCategory_H

#include "Lens.h"
#include "CG_ECS_MediumCompCategory.h"

class NDPairList;

class ECS_MediumCompCategory : public CG_ECS_MediumCompCategory
{
   public:
      ECS_MediumCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
};

#endif
