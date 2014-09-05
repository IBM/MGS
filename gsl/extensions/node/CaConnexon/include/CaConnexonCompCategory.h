#ifndef CaConnexonCompCategory_H
#define CaConnexonCompCategory_H

#include "Lens.h"
#include "CG_CaConnexonCompCategory.h"
#include "../../../../../nti/CountableModel.h"

class NDPairList;

class CaConnexonCompCategory : public CG_CaConnexonCompCategory, public CountableModel
{
   public:
      CaConnexonCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void count();
};

#endif
