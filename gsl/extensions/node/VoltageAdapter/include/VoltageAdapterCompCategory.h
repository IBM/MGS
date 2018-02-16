#ifndef VoltageAdapterCompCategory_H
#define VoltageAdapterCompCategory_H

#include "Lens.h"
#include "CG_VoltageAdapterCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class VoltageAdapterCompCategory : public CG_VoltageAdapterCompCategory,
   public CountableModel
{
   public:
      VoltageAdapterCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void count();
};

#endif
