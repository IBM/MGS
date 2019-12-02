#include "Lens.h"
#include "DNEdgeSetCompCategory.h"
#include "NDPairList.h"
#include "CG_DNEdgeSetCompCategory.h"
#include <math.h>
#include <ostream>

#define SHD getSharedMembers()
extern void setupFunctionTables(); 

DNEdgeSetCompCategory::DNEdgeSetCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
   : CG_DNEdgeSetCompCategory(sim, modelName, ndpList)
{
}
void DNEdgeSetCompCategory::initializeShared(RNG& rng)
{
#if defined(HAVE_GPU)
   udef_um_fncIndex.increaseSizeTo(_nodes.size());
   setupFunctionTables();
#endif
}
