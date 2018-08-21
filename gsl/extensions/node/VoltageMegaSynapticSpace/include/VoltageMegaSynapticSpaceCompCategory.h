#ifndef VoltageMegaSynapticSpaceCompCategory_H
#define VoltageMegaSynapticSpaceCompCategory_H

#include "Lens.h"
#include "CG_VoltageMegaSynapticSpaceCompCategory.h"

class NDPairList;

class VoltageMegaSynapticSpaceCompCategory : public CG_VoltageMegaSynapticSpaceCompCategory
{
   public:
      VoltageMegaSynapticSpaceCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
};

#endif
