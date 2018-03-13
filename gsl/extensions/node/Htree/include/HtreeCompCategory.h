#ifndef HtreeCompCategory_H
#define HtreeCompCategory_H

#include "Lens.h"
#include "CG_HtreeCompCategory.h"

class NDPairList;

class HtreeCompCategory : public CG_HtreeCompCategory
{
   public:
      HtreeCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
};

#endif
