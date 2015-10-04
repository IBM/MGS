#include <mpi.h>

#include "NeuroDevCommandLine.h"
#include "NeuronPartitioner.h"
#include "Tissue.h"
#include "SegmentForceAggregator.h"
#include "VolumeDecomposition.h"
#include "Decomposition.h"
#include "AllInSegmentSpace.h"
#include "FrontSegmentSpace.h"
#include "FrontLimitedSegmentSpace.h"
#include "SegmentKeySegmentSpace.h"
#include "ANDSegmentSpace.h"
#include "ORSegmentSpace.h"
#include "NOTSegmentSpace.h"
#include "NeuroDevTissueSlicer.h"
#include "AllInTouchSpace.h"
#include "Director.h"
#include "SegmentForceDetector.h"
#include "Communicator.h"
#include "Params.h"
#include "TissueGrowthSimulator.hpp"

#include "BG_AvailableMemory.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <errno.h>
#include <unistd.h>
#include <vector>
#include <utility>
#include <string>
#include <assert.h>
#include <fstream>
#include <sys/time.h>
#include <string.h>
#include <math.h>

#ifndef UAIX
//#define USING_CVC
#endif

#ifdef USING_CVC
extern void cvc(int, float *, float *, int *, bool);
extern void set_cvc_config(char *);
#endif

#define TRAJECTORY_TYPE 3

int main(int argc, char *argv[])
{
  int size = 0, rank = 0;
  
  MPI_Init(&argc, &argv);//Initialize MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  MPI_Barrier(MPI_COMM_WORLD);
  
  NeuroDevCommandLine commandLine;
  if (commandLine.parse(argc, argv) == false) {
    MPI_Finalize();
    exit(0);
  }
  
  bool clientConnect=commandLine.getClientConnect();
#ifdef USING_CVC
  if (clientConnect) set_cvc_config("./cvc.config");
#endif
  
  char inputFilename[256];
  int inputFilenameLength=commandLine.getInputFileName().length();
  strcpy(inputFilename, commandLine.getInputFileName().c_str());
  
  bool logTranslations=false;   // SET TO TRUE TO LOG NEURON TRANSLATIONS TO FILE
  bool logRotations=false;      // SET TO TRUE TO LOG NEURON ROTATIONS TO FILE
  
  Tissue* tissue = new Tissue(size, rank, logTranslations, logRotations);
  
  bool resample=commandLine.getResample();
  
  Communicator* communicator = new Communicator();
  Director* director = new Director(communicator);
  
  bool dumpResampledNeurons=( (commandLine.getOutputFormat()=="t" || commandLine.getOutputFormat()=="bt") ) ? true : false;
  
  int X=commandLine.getX();
  int Y=commandLine.getY();
  int Z=commandLine.getZ();
  
  int nSlicers = commandLine.getNumberOfSlicers();
  if(nSlicers == 0 || nSlicers>size) nSlicers = size;
  int nSegmentForceDetectors = commandLine.getNumberOfDetectors();
  if (nSegmentForceDetectors == 0 || nSegmentForceDetectors>size) nSegmentForceDetectors = size;
  
  NeuronPartitioner* neuronPartitioner = new NeuronPartitioner(rank, inputFilename, resample, dumpResampledNeurons, commandLine.getPointSpacing());

  Decomposition* decomposition=0;
  VolumeDecomposition* volumeDecomposition=0;
  char* ext=&inputFilename[inputFilenameLength-3];
  if (strcmp(ext, "bin")==0) {
    neuronPartitioner->partitionBinaryNeurons(nSlicers, nSegmentForceDetectors, tissue);
  }
  else {
    neuronPartitioner->partitionTextNeurons(nSlicers, nSegmentForceDetectors, tissue);
  }
  volumeDecomposition = new VolumeDecomposition(rank, NULL, nSegmentForceDetectors, tissue, X, Y, Z);
  decomposition=volumeDecomposition;

  assert( (nSlicers==size) || (nSegmentForceDetectors==size) );

  SegmentForceAggregator* segmentForceAggregator = new SegmentForceAggregator(rank, nSlicers, nSegmentForceDetectors, tissue);  

  Params params;
  char paramS[256];
  strcpy(paramS, commandLine.getParamFileName().c_str());
  params.readDevParams(paramS);

#ifdef VERBOSE
  if (rank==0) std::cerr<<"Max Branch Order = "<<tissue->getMaxBranchOrder()<<std::endl;
#endif

  AllInTouchSpace detectionTouchSpace;	// OBJECT CHOICE : PARAMETERIZABLE
  SegmentForceDetector *segmentForceDetector = new SegmentForceDetector(rank, nSlicers, nSegmentForceDetectors, commandLine.getNumberOfThreads(),
									&decomposition, &detectionTouchSpace, neuronPartitioner, &params);

  int maxIterations = commandLine.getMaxIterations();
  if (maxIterations<0) {
    std::cerr<<"max-iterations must be >= 0!"<<std::endl;
    MPI_Finalize();
    exit(0);
  }
   
  double Econ=commandLine.getEnergyCon();
  double dT=commandLine.getTimeStep();
  double E=0,dE=0,En=0;
 
  TissueGrowthSimulator TissueSim(size, rank, tissue, director, segmentForceDetector, segmentForceAggregator, &params, commandLine.getInitialFront()-1);

  AllInSegmentSpace allInSegmentSpace;

  FrontSegmentSpace frontSegmentSpace(TissueSim);		        // OBJECT CHOICE : PARAMETERIZABLE
  FrontLimitedSegmentSpace frontLimitedSegmentSpace(TissueSim);       	// OBJECT CHOICE : PARAMETERIZABLE
  
  std::vector<std::pair<std::string, unsigned int> > probeKey;
  probeKey.push_back(std::pair<std::string, unsigned int>(std::string("BRANCHTYPE"), TRAJECTORY_TYPE) );
  SegmentKeySegmentSpace gliaSegmentSpace(probeKey);
  NOTSegmentSpace notGliaSegmentSpace(&gliaSegmentSpace);

  ANDSegmentSpace coveredSegmentSpace(&frontLimitedSegmentSpace, &notGliaSegmentSpace);
  ANDSegmentSpace gliaOnFrontSegmentSpace(&frontSegmentSpace, &gliaSegmentSpace);
  ORSegmentSpace gliaOnFrontFrontLimitedSegmentSpace(&coveredSegmentSpace, &gliaOnFrontSegmentSpace);

  NeuroDevTissueSlicer* neuroDevTissueSlicer = new NeuroDevTissueSlicer(rank, nSlicers, nSegmentForceDetectors, 
									tissue, &decomposition,
									&frontSegmentSpace, &params, segmentForceDetector->getEnergy());

#ifdef VERBOSE
  if (rank==0) std::cerr<<"Maximum Front level = "<<TissueSim.getMaxFrontNumber()<<std::endl;
#endif
  bool attemptConnect=true;
  unsigned iteration=0;
  int nspheres=0;
  float *positions=0, *radii=0;
  int *types=0;
  double start, now, then;
  start=then=MPI_Wtime();
  director->addCommunicationCouple(neuroDevTissueSlicer, segmentForceDetector);
  volumeDecomposition->resetCriteria(&gliaOnFrontFrontLimitedSegmentSpace);
  neuroDevTissueSlicer->resetSegmentSpace(&coveredSegmentSpace);
  segmentForceDetector->updateCoveredSegments(true);
  director->iterate();
  segmentForceDetector->updateCoveredSegments(false);
  director->addCommunicationCouple(segmentForceDetector, segmentForceAggregator);
  bool grow = TissueSim.AdvanceFront() && maxIterations>0;
  neuroDevTissueSlicer->resetSegmentSpace(&frontSegmentSpace);    
  while (grow) {
#ifdef VERBOSE
    if (rank==0) printf("Front level %d", TissueSim.getFrontNumber());
    if (!grow && rank==0) printf(" <FINAL> ");
    if (rank==0) printf("\n"); 
#endif
    En = 0;
    iteration = 0;
    do {
#ifdef USING_CVC
      if (clientConnect) {
	tissue->getVisualizationSpheres(frontLimitedSegmentSpace, nspheres, positions, radii, types);
	cvc(nspheres, positions, radii, types, attemptConnect);
	attemptConnect=false;
      }
#endif
      /* computeForces is inside this front simulation step, which is equivalent
	 to an entire step through the Director's CommunicationCouple list */
      TissueSim.FrontSimulationStep(iteration,dT,E);
      dE = E - En;
      En = E;
      now=MPI_Wtime();
      if (rank==0 && iteration<maxIterations) std::cout<<"front = "
			    <<TissueSim.getFrontNumber()<<", begin = "
			    <<iteration<<", E = "
			    <<E<<", dE = "
			    <<dE<<", T = "
			    <<now<<", dT = "
			    <<now-then<<"."<<std::endl;
      then=now;
    } while( fabs(dE) > Econ && iteration < maxIterations);

    if (rank==0) std::cout<<"front = "
			  <<TissueSim.getFrontNumber()<<", end = "
			  <<iteration<<", E = "
			  <<E<<", dE = "
			  <<dE<<"."<<std::endl;
#ifdef USING_CVC
    attemptConnect=true;
#endif
    director->clearCommunicationCouples();
    director->addCommunicationCouple(neuroDevTissueSlicer, segmentForceDetector);
    volumeDecomposition->resetCriteria(&gliaOnFrontFrontLimitedSegmentSpace);
    neuroDevTissueSlicer->resetSegmentSpace(&coveredSegmentSpace);
    segmentForceDetector->updateCoveredSegments(true);
    director->iterate();
    segmentForceDetector->updateCoveredSegments(false);
    director->clearCommunicationCouples();
    director->addCommunicationCouple(neuroDevTissueSlicer, segmentForceDetector);
    director->addCommunicationCouple(segmentForceDetector, segmentForceAggregator);
    tissue->clearSegmentForces();     
    grow = TissueSim.AdvanceFront();
    neuroDevTissueSlicer->resetSegmentSpace(&frontSegmentSpace);    
  }
  volumeDecomposition->resetCriteria(&allInSegmentSpace);

  now=MPI_Wtime();
  if (rank==0) std::cerr<<"Compute Time : "<<now-start<<std::endl;
 
  FILE* tissueOutFile=0;
  if (commandLine.getOutputFormat()=="t" || commandLine.getOutputFormat()=="bt") {
   std::string outExtension(".developed");
   if (maxIterations>0) {
     tissue->outputTextNeurons(outExtension, 0, 0);     
   }
   if (commandLine.getOutputFileName()!="") {
     int nextToWrite=0, written=0, segmentsWritten=0, globalOffset=0;
     while (nextToWrite<size) {
       MPI_Allreduce((void*)&written, (void*)&nextToWrite, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);      
       MPI_Allreduce((void*)&segmentsWritten, (void*)&globalOffset, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);      
       if (nextToWrite==rank) {
	 if ( ( tissueOutFile = fopen(commandLine.getOutputFileName().c_str(), (rank==0) ? "wt" : "at") ) == NULL) {
	   printf("Could not open the output file %s!\n", commandLine.getOutputFileName().c_str());
	   MPI_Finalize();
	   exit(0);
	 }
	 segmentsWritten=tissue->outputTextNeurons(outExtension, tissueOutFile, globalOffset);
	 fclose(tissueOutFile);
	 written=1;
       }
     }
   }
  }

  delete communicator;
  delete director;
  delete segmentForceAggregator;
  delete volumeDecomposition;
  delete neuroDevTissueSlicer;
  delete segmentForceDetector;
  delete [] positions;
  delete [] radii;
  delete [] types;
  delete tissue;
  delete neuronPartitioner;
  
  MPI_Finalize();
}

