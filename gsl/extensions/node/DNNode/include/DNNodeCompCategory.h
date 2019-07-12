#ifndef DNNodeCompCategory_H
#define DNNodeCompCategory_H

#include "Lens.h"
#include "CG_DNNodeCompCategory.h"

class NDPairList;

class DNNodeCompCategory : public CG_DNNodeCompCategory
{
   public:
      DNNodeCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
};

#endif
