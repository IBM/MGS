#ifndef MegaSynapticCleftCompCategory_H
#define MegaSynapticCleftCompCategory_H

#include "Lens.h"
#include "CG_MegaSynapticCleftCompCategory.h"

class NDPairList;

class MegaSynapticCleftCompCategory : public CG_MegaSynapticCleftCompCategory
{
   public:
      MegaSynapticCleftCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
};

#endif
