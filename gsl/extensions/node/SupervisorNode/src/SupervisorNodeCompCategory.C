#include "Lens.h"
#include "SupervisorNodeCompCategory.h"
#include "NDPairList.h"
#include "CG_SupervisorNodeCompCategory.h"
#include <string>
#include <map>

#define PRELIM_STATE DBL_MAX
#define SHD getSharedMembers()
#define ITER getSimulation().getIteration()
#define IMG_SIZE (1*28*28)

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
  shuffleDeck(dataset.training_images.size(),rng);
  SHD.x.increaseSizeTo(IMG_SIZE);
  for (int i=0; i<IMG_SIZE; ++i) SHD.x[i]=PRELIM_STATE;
  SHD.imageIndex=-1;
  SHD.trainingPass=1;
}

void SupervisorNodeCompCategory::updateShared(RNG& rng) 
{
  SHD.refreshErrors = false;
  bool output=false;
  ++SHD.numberOfInputs;

  unsigned label, oldLabel;

  if (!SHD.shready) SHD.shready = isReady();
  else oldLabel=SHD.labels[SHD.labelIndex];
  
  if (!SHD.test) {
    do {
      if (++SHD.imageIndex==dataset.training_images.size()) {
	SHD.imageIndex=0;
	shuffleDeck(dataset.training_images.size(),rng);
	if (SHD.shready) output=true;
	if (++SHD.trainingPass>SHD.trainingIterations) {
	  SHD.test = true;
	  shuffleDeck(dataset.test_images.size(),rng);
	}
      }
      label = dataset.training_labels[_shuffledDeck[SHD.imageIndex]];
      if (SHD.shready) SHD.labels[SHD.labelIndex]=label;
    } while (label>SHD.numberOfLabels-1);
    for (unsigned idx=0; idx<IMG_SIZE; ++idx) {
      SHD.x[idx]=double(dataset.training_images[_shuffledDeck[SHD.imageIndex]][idx])/255.0;
    }
  }
  else {
    do {
      if (++SHD.imageIndex==dataset.test_images.size()) {
	SHD.imageIndex=0;
	shuffleDeck(dataset.test_images.size(),rng);
	if (SHD.shready) output=true;
      }
      label = dataset.test_labels[_shuffledDeck[SHD.imageIndex]];
      if (SHD.shready) SHD.labels[SHD.labelIndex]=label;
    } while (label>SHD.numberOfLabels-1);
    for (unsigned idx=0; idx<IMG_SIZE; ++idx)
      SHD.x[idx]=double(dataset.test_images[_shuffledDeck[SHD.imageIndex]][idx])/255.0;
  }
  if (!SHD.shready) {
    SHD.labels.push_back(label);
  }
  else {
    if (++SHD.labelIndex == SHD.labels.size()) SHD.labelIndex = 0;
  }
  if (output) outputError(oldLabel);
}

void SupervisorNodeCompCategory::outputError(unsigned currentLabel)
{
  double outError=0;
  ShallowArray<SupervisorNode>::iterator nodesIter=_nodes.begin(), nodesEnd=_nodes.end();
  for (unsigned rank=0; rank!=getSimulation().getNumProcesses(); ++rank) {
    if (getSimulation().getRank()==rank) {
      for (; nodesIter!=nodesEnd; ++nodesIter) {
	outError += nodesIter->sumOfSquaredError;
	std::cerr<<getSimulation().getRank()<<" : "
		 <<currentLabel<<" : "
		 <<( (currentLabel==nodesIter->getGlobalIndex()) ? 1.0 : 0.0)
		 <<"   |   "<<*(nodesIter->predictions)[nodesIter->getGlobalIndex()]
		 <<"   |   "<<(nodesIter->logits)[nodesIter->getGlobalIndex()]
		 <<std::endl<<std::flush;
       }
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  double currentError=0;
  MPI_Allreduce(&outError, &currentError, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  currentError/=(SHD.numberOfInputs*SHD.numberOfLabels);
  currentError=sqrt(currentError);
  if (getSimulation().getRank()==0) {
    std::cout<<ITER<<" : current error = "<<currentError;
    std::cout<<" : wins ratio = "<<double(_nodes.begin()->wins)/double(SHD.numberOfInputs)<<std::endl<<std::flush;
  }
  SHD.numberOfInputs = 0;
  SHD.refreshErrors = true;
}

bool SupervisorNodeCompCategory::isReady()
{
  bool rval=true;
  ShallowArray<SupervisorNode>::iterator nodesIter, nodesEnd=_nodes.end();
  for (nodesIter=_nodes.begin(); nodesIter!=nodesEnd; ++nodesIter) {
    rval=nodesIter->ready;
    if (!rval) break;
  }
  bool globalRval=false;
  MPI_Allreduce(&rval, &globalRval, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
  return globalRval;
}

void SupervisorNodeCompCategory::shuffleDeck(unsigned deckSize, RNG& rng)
{
  _shuffledDeck.clear();
  std::map<double, unsigned> shuffler;
  std::map<double, unsigned>::iterator miter;
  double d;
  for (unsigned i=0; i<deckSize; ++i) {
    do {
      d = drandom(rng);
      miter = shuffler.find(d);
    } while (miter!=shuffler.end());
    shuffler[d]=i;
  }
  std::map<double, unsigned>::iterator mend=shuffler.end();
  for (miter=shuffler.begin(); miter!=mend; ++miter) {
    _shuffledDeck.push_back(miter->second);
  }
}
