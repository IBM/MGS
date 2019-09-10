#include "Lens.h"
#include "DNNodeCompCategory.h"
#include "NDPairList.h"
#include "CG_DNNodeCompCategory.h"

#define SHD getSharedMembers()

DNNodeCompCategory::DNNodeCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
   : CG_DNNodeCompCategory(sim, modelName, ndpList)
{
}
