#ifndef IzhikUnitCompCategory_H
#define IzhikUnitCompCategory_H

#include "Lens.h"
#include "CG_IzhikUnitCompCategory.h"

class NDPairList;

class IzhikUnitCompCategory : public CG_IzhikUnitCompCategory
{
   public:
      IzhikUnitCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
};

#endif
