#include "Lens.h"
#include "DNEdgeSetCompCategory.h"
#include "NDPairList.h"
#include "CG_DNEdgeSetCompCategory.h"
#include <math.h>

#define SHD getSharedMembers()

DNEdgeSetCompCategory::DNEdgeSetCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
   : CG_DNEdgeSetCompCategory(sim, modelName, ndpList)
{
}
