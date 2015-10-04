#include <mpi.h>

#define SLICING_DIMENSIONS 3 // for now, this must be 3
//#define SYNAPSE_PARAMS_TOUCH_DETECT

#include "MaxComputeOrder.h"
#include "NeuroDevCommandLine.h"
#include "NeuronPartitioner.h"
#include "Tissue.h"
#include "PassThroughTouchFilter.h"
#include "AntiredundancyTouchFilter.h"
#include "TouchAggregator.h"
#include "VolumeDecomposition.h"
#include "Decomposition.h"
#include "AllInSegmentSpace.h"
#include "AllInTouchSpace.h"
#include "TouchDetectTissueSlicer.h"
#include "SynapseTouchSpace.h"
#include "ORTouchSpace.h"
#include "TouchAnalyzer.h"
#include "ZeroTouchAnalysis.h"
#include "TouchTable.h"
#include "Params.h"

#include "Director.h"
#include "TouchDetector.h"
#include "Communicator.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <errno.h>
#include <unistd.h>
#include <vector>
#include <assert.h>
#include <fstream>
#include <sys/time.h>
#include <string.h>
#include <math.h>

inline static double AvailableMemory(void);

//Returns the amount of free memory, in Mbytes
static double AvailableMemory(void)
{
#ifdef __blrts__
  double MBytes;
  unsigned long val, st[2];

  st[0] = 123456;
  val = (unsigned long)st;
  MBytes = (double)(val - (unsigned long)sbrk(0)) * 0.00000095367431640625;
  return (MBytes);
#else
  return 0;
#endif /* __blrts__ */
}

int main(int argc, char *argv[])
{
   double timeStep0, timeStep1, timeStep2, timeStep3, timeStep4, memorySize1, memorySize2;;
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

   timeStep0 = MPI_Wtime();
   char inputFilename[256];
   int inputFilenameLength=commandLine.getInputFileName().length();
   strcpy(inputFilename, commandLine.getInputFileName().c_str());

   bool logTranslations=false;   // SET TO TRUE TO LOG NEURON TRANSLATIONS TO FILE
   bool logRotations=false;      // SET TO TRUE TO LOG NEURON ROTATIONST TO FILE

   Tissue* tissue = new Tissue(size, rank, logTranslations, logRotations);

   bool resample=commandLine.getResample();

   Communicator* communicator = new Communicator();   
   Director* director = new Director(communicator);

   int nSlicers = commandLine.getNumberOfSlicers();
   if(nSlicers == 0 || nSlicers>size) nSlicers = size;

   int nTouchDetectors = commandLine.getNumberOfDetectors();
   if (nTouchDetectors == 0) nTouchDetectors = size;

   int X=commandLine.getX();
   int Y=commandLine.getY();
   int Z=commandLine.getZ();  

   bool dumpResampledNeurons=(commandLine.getOutputFormat()=="t") ? true : false;
   NeuronPartitioner* neuronPartitioner = new NeuronPartitioner(rank, inputFilename, resample, dumpResampledNeurons, commandLine.getPointSpacing());
   VolumeDecomposition* volumeDecomposition=0;
   Decomposition* decomposition=0;
   char* ext=&inputFilename[inputFilenameLength-3];
   if (strcmp(ext, "bin")==0) {
     neuronPartitioner->partitionBinaryNeurons(nSlicers, nTouchDetectors, tissue);
   }
   else if (strcmp(ext, "txt")==0) {
     neuronPartitioner->partitionTextNeurons(nSlicers, nTouchDetectors, tissue);
   }
   else {
     std::cerr<<"Unrecognized input file extension: "<<&inputFilename[inputFilenameLength-3]<<std::endl;
     exit(0);
   }
	
   assert( (nSlicers==size) || (nTouchDetectors==size) );

   Params params;
   char paramS[256];
   strcpy(paramS, commandLine.getParamFileName().c_str());
   params.readDetParams(paramS);
   
   bool autapses=false;
  
   std::ifstream testParamFile("SynParams.par");
   if (!testParamFile) {
     std::cerr<<"Stand alone touch detection requires SynParams.par file!"<<std::endl;
     exit(0);
   }
   testParamFile.close();
   params.readSynParams("SynParams.par");

   SynapseTouchSpace electricalSynapseTouchSpace(SynapseTouchSpace::ELECTRICAL,
						 &params,
						 autapses);
  
   SynapseTouchSpace chemicalSynapseTouchSpace(SynapseTouchSpace::CHEMICAL,
					       &params,
					       autapses);
  
   ORTouchSpace detectionSpace(electricalSynapseTouchSpace, chemicalSynapseTouchSpace);

   TouchSpace* communicateTouchSpace = new AllInTouchSpace;
   TouchAggregator* touchAggregator = 0;
   //touchAggregator = new TouchAggregator(rank, nTouchDetectors, communicateTouchSpace, 0);  // IF CREATED, TOUCHES ARE AGGREGATED
   TouchFilter* touchFilter = 0;
   //touchFilter = new AntiredundancyTouchFilter(touchAggregator, tissue);
   //touchFilter = new PassThroughTouchFilter(touchAggregator, tissue);

   timeStep1 = MPI_Wtime();

   volumeDecomposition = new VolumeDecomposition(rank, NULL, nTouchDetectors, tissue, X, Y, Z);
   decomposition=volumeDecomposition;

   TissueSlicer* tissueSlicer = new TouchDetectTissueSlicer(rank, nSlicers, nTouchDetectors, 
							    tissue, &decomposition, 0, &params, MAX_COMPUTE_ORDER);
   
   
   TouchDetector *touchDetector = new TouchDetector(rank, nSlicers, nTouchDetectors, MAX_COMPUTE_ORDER, commandLine.getNumberOfThreads(),
						    commandLine.getAppositionSamplingRate(), &decomposition, 
						    &detectionSpace, communicateTouchSpace, neuronPartitioner, 
						    0, &params);

   int maxIterations = commandLine.getMaxIterations();
   if (maxIterations<=0) {
     std::cerr<<"max-iterations must be > 0!"<<std::endl;
     exit(0);
   }
   
   TouchAnalyzer* touchAnalyzer = new TouchAnalyzer(rank, commandLine.getExperimentName(), nSlicers, nTouchDetectors, 
						    touchDetector, tissue, maxIterations, 
						    touchFilter, true, true);

   std::vector<std::vector<SegmentDescriptor::SegmentKeyData> > touchTableMasks;
   params.getTouchTableMasks(touchTableMasks);
   std::vector<std::vector<SegmentDescriptor::SegmentKeyData> >::iterator ttiter=touchTableMasks.begin(),
     ttend=touchTableMasks.end();
   for (; ttiter!=ttend; ++ttiter) {
     TouchTable* touchTable = new TouchTable(*ttiter);
     touchTable->setOutput(false);
     touchAnalyzer->addTouchTable(touchTable);    // LIST CREATION : PARAMETERIZABLE : ANALYSIS-INDEPENDENT TOUCH TABLE
   }
   TouchAnalysis* touchAnalysis = new ZeroTouchAnalysis(tissue);
   touchAnalyzer->addTouchAnalysis(touchAnalysis);    // LIST CREATION : PARAMETERIZABLE

   timeStep2 = MPI_Wtime();
#ifdef SYNAPSE_PARAMS_TOUCH_DETECT
   decomposition->resetCriteria(&detectionSpace);
#endif
   director->addCommunicationCouple(tissueSlicer, touchDetector);
   //director->addCommunicationCouple(touchDetector, touchAggregator);
   director->addCommunicationCouple(touchAnalyzer, touchAnalyzer);
   timeStep3 = MPI_Wtime();

   memorySize1 = AvailableMemory();
   unsigned int iteration = 0;
   bool halt = false;
   while (!halt) {
     ++iteration;
     if (rank==0) printf("Iteration %d", iteration);
     if (touchAnalyzer->isDone()) {
       halt = true;
       if (rank==0) printf(" <FINAL> ");
     }
     if (rank==0) printf("\n");
	 
     director->iterate();

     memorySize2 = AvailableMemory();
	
     touchAnalyzer->analyze(iteration);
     timeStep4 = MPI_Wtime();

     /*	 
     if (rank==0) printf("io=%.3f slice=%.3f comm=%.3f tDetection=%.3f tTotal=%.3f MemBefore=%.2f MemAfter=%.2f\n",
			 timeStep1 - timeStep0,
			 timeStep2 - timeStep1,
			 timeStep3 - timeStep2,
			 timeStep4 - timeStep3,
			 timeStep4 - timeStep0,
			 memorySize1,memorySize2);
     */
   }
   delete communicator;
   delete director;
   delete tissue;
   delete neuronPartitioner;
   delete communicateTouchSpace;
   delete touchAggregator;
   delete touchFilter;
   delete volumeDecomposition;
   delete tissueSlicer;
   delete touchAnalyzer; 
   delete touchDetector;

   MPI_Finalize();
}

