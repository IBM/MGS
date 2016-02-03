#ifndef CaERConcentrationEndPointCompCategory_H
#define CaERConcentrationEndPointCompCategory_H

#include "Lens.h"
#include "CG_CaERConcentrationEndPointCompCategory.h"

class NDPairList;

class CaERConcentrationEndPointCompCategory : public CG_CaERConcentrationEndPointCompCategory 
{
   public:
      CaERConcentrationEndPointCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
};

#endif
