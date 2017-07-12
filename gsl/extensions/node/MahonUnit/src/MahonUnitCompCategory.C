#include "Lens.h"
#include "MahonUnitCompCategory.h"
#include "NDPairList.h"
#include "CG_MahonUnitCompCategory.h"

#define ITER getSimulation().getIteration()
#define SHD getSharedMembers()




MahonUnitCompCategory::MahonUnitCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
   : CG_MahonUnitCompCategory(sim, modelName, ndpList)
{
}
