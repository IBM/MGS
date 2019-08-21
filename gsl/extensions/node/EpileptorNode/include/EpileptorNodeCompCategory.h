#ifndef EpileptorNodeCompCategory_H
#define EpileptorNodeCompCategory_H

#include "Lens.h"
#include "CG_EpileptorNodeCompCategory.h"

class NDPairList;

class EpileptorNodeCompCategory : public CG_EpileptorNodeCompCategory
{
   public:
      EpileptorNodeCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      std::map<std::pair<int, int>, float> connectionMap;
};

#endif
