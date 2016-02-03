#include "Lens.h"
#include "CaERConcentrationCompCategory.h"
#include "NDPairList.h"
#include "CG_CaERConcentrationCompCategory.h"

CaERConcentrationCompCategory::CaERConcentrationCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
   : CG_CaERConcentrationCompCategory(sim, modelName, ndpList)
{
}

void CaERConcentrationCompCategory::deriveParameters(RNG& rng) 
{
	//TUAN: TODO
  if (getSharedMembers().deltaT) {
    getSharedMembers().bmt = 2.0 / (getSharedMembers().beta * *(getSharedMembers().deltaT)) ;
#ifdef DEBUG_HH
    std::cerr<<getSimulation().getRank()<<" : CaERConcentrationes : "<<_nodes.size()<<" [ ";
    for (int i=0; i<_nodes.size(); ++i)
      std::cerr<<_nodes[i].getSize()<<" ";
    std::cerr<<" ]"<<std::endl;
#endif
  }
}

void CaERConcentrationCompCategory::count() 
{
  long long totalCount, localCount=_nodes.size();
  MPI_Allreduce((void*) &localCount, (void*) &totalCount, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  float localVar, totalVar, mean=float(totalCount)/getSimulation().getNumProcesses();
  localVar=(float(localCount)-mean)*(float(localCount)-mean);
  MPI_Allreduce((void*) &localVar, (void*) &totalVar, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  float std=sqrt(totalVar/getSimulation().getNumProcesses());
  if (getSimulation().getRank()==0) printf("Total CaERConcentration = %lld, Mean = %lf, StDev = %lf\n", totalCount, mean, std);

  ////

  totalCount=localCount=0;
  ShallowArray<CaERConcentration>::iterator nodesIter=_nodes.begin(), nodesEnd=_nodes.end();
  for (; nodesIter!=nodesEnd; ++nodesIter) {
    localCount+=nodesIter->dimensions.size();
  }

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Allreduce((void*) &localCount, (void*) &totalCount, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  mean=totalCount/getSimulation().getNumProcesses();
  localVar=(float(localCount)-mean)*(float(localCount)-mean);
  MPI_Allreduce((void*) &localVar, (void*) &totalVar, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  std=sqrt(totalVar/getSimulation().getNumProcesses());
  if (getSimulation().getRank()==0) printf("Total CaER Compartment = %lld, Mean = %lf, StDev = %lf\n", totalCount, mean, std);
}
