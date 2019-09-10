#ifndef DNEdgeSetCompCategory_H
#define DNEdgeSetCompCategory_H

#include "Lens.h"
#include "CG_DNEdgeSetCompCategory.h"
#include "TransferFunction.h"

class NDPairList;

class DNEdgeSetCompCategory : public CG_DNEdgeSetCompCategory
{
   public:
      DNEdgeSetCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
};

#endif
