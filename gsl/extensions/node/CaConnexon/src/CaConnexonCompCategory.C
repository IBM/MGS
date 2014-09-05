#include "Lens.h"
#include "CaConnexonCompCategory.h"
#include "NDPairList.h"
#include "CG_CaConnexonCompCategory.h"

CaConnexonCompCategory::CaConnexonCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
   : CG_CaConnexonCompCategory(sim, modelName, ndpList)
{
}

void CaConnexonCompCategory::count() 
{
  long long totalCount, localCount=_nodes.size();
  MPI_Allreduce((void*) &localCount, (void*) &totalCount, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  float localVar, totalVar, mean=float(totalCount)/getSimulation().getNumProcesses();
  localVar=(float(localCount)-mean)*(float(localCount)-mean);
  MPI_Allreduce((void*) &localVar, (void*) &totalVar, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  float std=sqrt(totalVar/getSimulation().getNumProcesses());
  if (getSimulation().getRank()==0) printf("Total CaConnexon = %lld, Mean = %lf, StDev = %lf\n", totalCount, mean, std);
}

