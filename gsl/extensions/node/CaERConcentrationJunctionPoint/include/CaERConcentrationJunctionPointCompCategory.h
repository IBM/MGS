#ifndef CaERConcentrationJunctionPointCompCategory_H
#define CaERConcentrationJunctionPointCompCategory_H

#include "Lens.h"
#include "CG_CaERConcentrationJunctionPointCompCategory.h"

class NDPairList;

class CaERConcentrationJunctionPointCompCategory : public CG_CaERConcentrationJunctionPointCompCategory
{
   public:
      CaERConcentrationJunctionPointCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
};

#endif
