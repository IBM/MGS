#ifndef NVUNodeCompCategory_H
#define NVUNodeCompCategory_H

#include "Lens.h"
#include "CG_NVUNodeCompCategory.h"

class NDPairList;

class NVUNodeCompCategory : public CG_NVUNodeCompCategory
{
   public:
      NVUNodeCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void paramInitialize(RNG& rng);

};

#endif
