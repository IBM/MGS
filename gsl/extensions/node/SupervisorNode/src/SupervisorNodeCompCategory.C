#include "Lens.h"
#include "SupervisorNodeCompCategory.h"
#include "NDPairList.h"
#include "CG_SupervisorNodeCompCategory.h"
#include <string>

#define SHD getSharedMembers()
#define ITER getSimulation().getIteration()

SupervisorNodeCompCategory::SupervisorNodeCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
   : CG_SupervisorNodeCompCategory(sim, modelName, ndpList)
{
}

void SupervisorNodeCompCategory::initializeShared(RNG& rng) 
{
  dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>
    (std::string(SHD.dataLocation.c_str()));
  std::cerr<<"mnist_reader loaded "<<dataset.training_images.size()<<" training images."<<std::endl;
  std::cerr<<"mnist_reader loaded "<<dataset.test_images.size()<<" test images."
	   <<std::endl<<std::endl;
  SHD.x.increaseSizeTo(1 * 28 * 28);
  SHD.trainingPass=1;
}

void SupervisorNodeCompCategory::updateShared(RNG& rng) 
{
  SHD.refreshErrors = false;
  if (!SHD.test) {
    for (unsigned idx=0; idx<(1 * 28 * 28); ++idx) {
      assert(dataset.training_images.size()>SHD.imageIndex);
      assert(dataset.training_images[SHD.imageIndex].size()>idx);
      assert(SHD.x.size()>idx);
      SHD.x[idx]=double(dataset.training_images[SHD.imageIndex][idx])/255.0;
    }
    SHD.label = dataset.training_labels[SHD.imageIndex];
    
    if (++SHD.imageIndex==dataset.training_images.size()) {      
      SHD.imageIndex=0;
      outputError();
      if (++SHD.trainingPass>SHD.trainingIterations)
	SHD.test = true;
    }
  }
  else {
    for (unsigned idx=0; idx<(1 * 28 * 28); ++idx) {
      SHD.x[idx]=double(dataset.test_images[SHD.imageIndex][idx])/255.0;
    }
    SHD.label = dataset.test_labels[SHD.imageIndex];
    if (++SHD.imageIndex==dataset.test_images.size())
      outputError();
  }
}

void SupervisorNodeCompCategory::outputError()
{
  SHD.refreshErrors = true;
  double outError=0;
  ShallowArray<SupervisorNode>::iterator nodesIter=_nodes.begin(), nodesEnd=_nodes.end();
  for (; nodesIter!=nodesEnd; ++nodesIter) {
    outError += nodesIter->sumOfSquaredError;
    std::cerr<<((SHD.label==nodesIter->getGlobalIndex()) ? 1.0 : 0)
	     <<"   |   "<<*(nodesIter->prediction)<<std::endl;
  }
  double currentError=0;
  MPI_Allreduce(&outError, &currentError, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  currentError/=_nodes.size();
  if (getSimulation().getRank()==0)
    std::cout<<ITER<<" : current error = "<<currentError<<std::endl;
}
