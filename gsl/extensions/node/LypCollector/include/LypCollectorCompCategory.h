#ifndef LypCollectorCompCategory_H
#define LypCollectorCompCategory_H

#include "Lens.h"
#include "CG_LypCollectorCompCategory.h"

class NDPairList;

class LypCollectorCompCategory : public CG_LypCollectorCompCategory
{
   public:
      LypCollectorCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
};

#endif
