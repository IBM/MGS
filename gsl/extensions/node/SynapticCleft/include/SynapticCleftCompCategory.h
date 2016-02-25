#ifndef SynapticCleftCompCategory_H
#define SynapticCleftCompCategory_H

#include "Lens.h"
#include "CG_SynapticCleftCompCategory.h"

class NDPairList;

class SynapticCleftCompCategory : public CG_SynapticCleftCompCategory
{
   public:
      SynapticCleftCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
};

#endif
