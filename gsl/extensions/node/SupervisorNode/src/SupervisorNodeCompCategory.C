#include "Lens.h"
#include "SupervisorNodeCompCategory.h"
#include "NDPairList.h"
#include "CG_SupervisorNodeCompCategory.h"
#include <string>
#include <map>

#define PRELIM_STATE DBL_MAX
#define SHD getSharedMembers()
#define ITER getSimulation().getIteration()

/* NOTE: MNIST-specific data information */
#define IMG_SIZE (1*28*28)

//#define DONT_SHUFFLE

//#define DEBUG_LOAD_SINGLE_IMAGE

//#define LOAD_ALL_IMAGES_IN_GPU

SupervisorNodeCompCategory::SupervisorNodeCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
   : CG_SupervisorNodeCompCategory(sim, modelName, ndpList)
{
}
SupervisorNodeCompCategory::~SupervisorNodeCompCategory()
{
#ifdef HAVE_GPU
  if (allocated_gpu)
  {
    cudaFree(d_buffer);
    cudaFree(training_images);
    cudaFree(test_images);
    cudaFree(l_buffer);
  }
#endif
}

void SupervisorNodeCompCategory::initializeShared(RNG& rng) 
{
#ifdef HAVE_GPU
  /* add for GPU */
  udef_um_globalIdx.increaseSizeTo(_nodes.size());
  /* must be pre-allocated to work on GPU */
  int ii = 0;
  for (auto iter=_nodes.begin(); iter != _nodes.end(); iter++, ii++)
    udef_um_globalIdx[ii] = iter->getGlobalIndex();
#endif
  dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>
    (std::string(SHD.dataLocation.c_str()));
  std::cerr<<"mnist_reader loaded "<<dataset.training_images.size()<<" training images."<<std::endl;
  std::cerr<<"mnist_reader loaded "<<dataset.test_images.size()<<" test images."
	   <<std::endl<<std::endl;

#ifdef HAVE_GPU
#if defined(LOAD_ALL_IMAGES_IN_GPU)
  loadDataToGPU();
#endif
#endif

  shuffleDeck(dataset.training_images.size(),rng);
  SHD.x->increaseSizeTo(IMG_SIZE);
  for (int i=0; i<IMG_SIZE; ++i) (*(SHD.x))[i]=PRELIM_STATE;
  SHD.imageIndex=-1;
  SHD.trainingEpoch=1;
}

#ifdef HAVE_GPU
void SupervisorNodeCompCategory::loadDataToGPU()
{
  allocated_gpu = true;
  count_train = dataset.training_images.size();
  count_test = dataset.test_images.size();
  size_t size = (count_train + count_test) * IMG_SIZE*sizeof(double);
  cudaMallocManaged(&d_buffer, size);
  for (int ii=0; ii < count_train; ii++)
  {
    int idx = ii*IMG_SIZE;
    for (int jj=0; jj < IMG_SIZE; jj++)
    {
      d_buffer[idx+jj]=double(dataset.training_images[ii][jj])/255.0;
    }
  }
  int offset = count_train * IMG_SIZE;
  for (int ii=0; ii < count_test; ii++)
  {
    int idx = ii*IMG_SIZE;
    for (int jj=0; jj < IMG_SIZE; jj++)
    {
      d_buffer[offset + idx+jj]=double(dataset.test_images[ii][jj])/255.0;
    }
  }
  cudaMallocManaged(&training_images, count_train * sizeof(double*));
  cudaMallocManaged(&test_images, count_test * sizeof(double*));
  for (int ii=0; ii < count_train; ii++)
    training_images[ii] = d_buffer+(ii*IMG_SIZE);
  offset = count_train;
  for (int ii=0; ii < count_test; ii++)
    test_images[ii] = d_buffer+((offset+ii)*IMG_SIZE);

  cudaMallocManaged(&l_buffer, (count_train + count_test) * sizeof(uint8_t));
  training_labels = l_buffer+(0);
  test_labels = l_buffer+(count_train);
}
#endif

void SupervisorNodeCompCategory::updateShared(RNG& rng) 
{
#if defined(HAVE_GPU) && defined(LOAD_ALL_IMAGES_IN_GPU)
  updateShared_GPU(rng);
#else
  updateShared_origin(rng);
#endif
}
void SupervisorNodeCompCategory::updateShared_origin(RNG& rng) 
{
  SHD.refreshErrors = false;
  bool output=false;
  ++SHD.numberOfInputs; //number of inputs (e.g. images) used so far

  unsigned label, oldLabel;

  if (!SHD.shready) SHD.shready = isReady();
  else oldLabel=SHD.labels[SHD.labelIndex];
  
  if (!SHD.test) {
    do {
      if (++SHD.imageIndex==dataset.training_images.size()) {
	SHD.imageIndex=0;
	shuffleDeck(dataset.training_images.size(),rng);
	if (SHD.shready) output=true;
	if (++SHD.trainingEpoch>SHD.trainingEpochs) {
	  SHD.test = true;
	  shuffleDeck(dataset.test_images.size(),rng);
	}
      }
      label = dataset.training_labels[_shuffledDeck[SHD.imageIndex]];
      if (SHD.shready) SHD.labels[SHD.labelIndex]=label;
    } while (label>SHD.numberOfLabels-1);
    for (unsigned idx=0; idx<IMG_SIZE; ++idx) {
      (*(SHD.x))[idx]=double(dataset.training_images[_shuffledDeck[SHD.imageIndex]][idx])/255.0;
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
      (*(SHD.x))[idx]=double(dataset.test_images[_shuffledDeck[SHD.imageIndex]][idx])/255.0;
  }
  if (!SHD.shready) {
    SHD.labels.push_back(label); //keep the label until it is being used to calculate gradient - eventually it's the size of the DNN
  }
  else {
    if (++SHD.labelIndex == SHD.labels.size()) SHD.labelIndex = 0;
  }
  if (output) outputError(oldLabel);
}

#ifdef HAVE_GPU
void SupervisorNodeCompCategory::updateShared_GPU(RNG& rng) 
{
  SHD.refreshErrors = false;
  bool output=false;
  ++SHD.numberOfInputs; //number of inputs (e.g. images) used so far

  unsigned label, oldLabel;

  if (!SHD.shready) SHD.shready = isReady();
  else oldLabel=SHD.labels[SHD.labelIndex];
  
  if (!SHD.test) {
    do {
      if (++SHD.imageIndex == count_train) {
	SHD.imageIndex=0;
	shuffleDeck(count_train, rng);
	if (SHD.shready) output=true;
	if (++SHD.trainingEpoch>SHD.trainingEpochs) {
	  SHD.test = true;
	  shuffleDeck(count_test, rng);
	}
      }
      label = dataset.training_labels[_shuffledDeck[SHD.imageIndex]];
      if (SHD.shready) SHD.labels[SHD.labelIndex]=label;
    } while (label>SHD.numberOfLabels-1);
    bool delete_current_data = false;
    SHD.x->changeRef(training_images[_shuffledDeck[SHD.imageIndex]], IMG_SIZE, delete_current_data);
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
    bool delete_current_data = false;
    SHD.x->changeRef(test_images[_shuffledDeck[SHD.imageIndex]], IMG_SIZE, delete_current_data);
  }
  if (!SHD.shready) {
    SHD.labels.push_back(label); //keep the label until it is being used to calculate gradient - eventually it's the size of the DNN
  }
  else {
    if (++SHD.labelIndex == SHD.labels.size()) SHD.labelIndex = 0;
  }
  if (output) outputError(oldLabel);
}
#endif

void SupervisorNodeCompCategory::outputError(unsigned currentLabel)
{
  double outError=0;
  auto nodesEnd=_nodes.end();
  for (unsigned rank=0; rank!=getSimulation().getNumProcesses(); ++rank) {
    if (getSimulation().getRank()==rank) {
      for (auto nodesIter=_nodes.begin(); nodesIter!=nodesEnd; ++nodesIter) {
#if defined(HAVE_GPU)
        outError += nodesIter->_container->um_sumOfSquaredError[nodesIter->__index__];
#else
	outError += nodesIter->sumOfSquaredError;
#endif
#if defined(HAVE_GPU)
	{
        std::cerr<<getSimulation().getRank()<<" : "
                 <<currentLabel<<" : "
                 <<( (currentLabel==nodesIter->getGlobalIndex()) ? 1.0 : 0.0)
                 <<"   |   "<<*(nodesIter->_container->um_logits[nodesIter->__index__])[nodesIter->getGlobalIndex()]
                 <<"   |   "<<(nodesIter->_container->um_predictions[nodesIter->__index__])[nodesIter->getGlobalIndex()]
                 <<std::endl<<std::flush;
	}
#else
	std::cerr<<getSimulation().getRank()<<" : "
		 <<currentLabel<<" : "
		 <<( (currentLabel==nodesIter->getGlobalIndex()) ? 1.0 : 0.0)
		 <<"   |   "<<*(nodesIter->logits)[nodesIter->getGlobalIndex()]
		 <<"   |   "<<(nodesIter->predictions)[nodesIter->getGlobalIndex()]
		 <<std::endl<<std::flush;
#endif
       }
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  double currentError=0;
  MPI_Allreduce(&outError, &currentError, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  currentError/=(SHD.numberOfInputs*SHD.numberOfLabels);
  currentError=sqrt(currentError);
  if (getSimulation().getRank()==0) {
    std::cout<<ITER<<" : current error = "<< currentError;
    //every SupevisorNode does the same calculation so we only needs to get 'wins' from the first SupervisorNode
#if defined(HAVE_GPU)
    std::cout<<" : wins ratio = "<<double(_nodes.begin()->_container->um_wins[_nodes.begin()->__index__])/double(SHD.numberOfInputs);
#else
    std::cout<<" : wins ratio = "<<double(_nodes.begin()->wins)/double(SHD.numberOfInputs);
#endif
    std::cout << " time passed: " << getSimulation().getTimer().lapWallTime();
    std::cout <<std::endl<<std::flush;
  }
  SHD.numberOfInputs = 0;
  SHD.refreshErrors = true;
}

bool SupervisorNodeCompCategory::isReady()
{
  bool rval=true;
  auto nodesEnd=_nodes.end();
  for (auto nodesIter=_nodes.begin(); nodesIter!=nodesEnd; ++nodesIter) {
#if defined(HAVE_GPU)
    rval=nodesIter->_container->um_ready[nodesIter->__index__];
#else
    rval=nodesIter->ready;
#endif
    if (!rval) break;
  }
  bool globalRval=false;
  MPI_Allreduce(&rval, &globalRval, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
  return globalRval;
}

void SupervisorNodeCompCategory::shuffleDeck(unsigned deckSize, RNG& rng)
{
  if (_shuffledDeck.size() != deckSize)
    _shuffledDeck.increaseSizeTo(deckSize);
#ifdef DONT_SHUFFLE
  unsigned ii=0;
  for (unsigned ii=0; ii < deckSize; ++ii) {
    _shuffledDeck[ii] = ii;
  }
#else
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
  unsigned ii=0;
  for (miter=shuffler.begin(); miter!=mend; ++miter, ++ii) {
    _shuffledDeck[ii] = miter->second;
  }
#endif

#ifdef HAVE_GPU
  {///GPU-specific
    if (um_shuffledDeck.size() != deckSize)
      um_shuffledDeck.increaseSizeTo(deckSize);
    auto mend=_shuffledDeck.end();
    int ii =0;
    for (auto miter=_shuffledDeck.begin(); miter!=mend; ++miter, ++ii) {
      um_shuffledDeck[ii] = (*miter);
    }
  }
#endif
}
