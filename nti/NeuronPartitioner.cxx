// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. and EPFL 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// ================================================================

#include "NeuronPartitioner.h"
#include "Tissue.h"
#include "Branch.h"
#include "Segment.h"
#include "Params.h"
#include <map>
#include <cassert>

NeuronPartitioner::NeuronPartitioner(int rank, char* inputFileName, bool resample, bool dumpOutput, double pointSpacing) :
  _nSlicers(0),
  _size(0),
  _rank(rank),
  _totalNeurons(0),
  _totalSegmentsRead(0),
  _logTranslationHistory(false),
  _logRotationHistory(false),
  _endNeurons(0),
  _neuronGroupNeuronCount(0),
  _neuronGroupSegCount(0),
  _neuronGroupStartNeuron(0),
  _neuronSegments(0),
  _resample(resample),
  _dumpOutput(dumpOutput),
  _pointSpacing(pointSpacing)
{
  if (_pointSpacing<1.0) {
    if (_rank==0) std::cerr<<"Warning: Geometric spacing factor reset to 1.0!"<<std::endl;
    _pointSpacing=1.0;
  }
  strcpy(_inputFilename, inputFileName);
}

NeuronPartitioner::NeuronPartitioner(NeuronPartitioner& np) :
  _nSlicers(np._nSlicers),
  _size(np._size),
  _rank(np._rank),
  _totalNeurons(np._totalNeurons),
  _totalSegmentsRead(np._totalSegmentsRead),
  _logTranslationHistory(np._logTranslationHistory),
  _logRotationHistory(np._logRotationHistory),
  _endNeurons(0),
  _neuronGroupNeuronCount(np._neuronGroupNeuronCount),
  _neuronGroupSegCount(np._neuronGroupSegCount),
  _neuronGroupStartNeuron(np._neuronGroupStartNeuron),
  _neuronSegments(0),
  _resample(np._resample),
  _dumpOutput(np._dumpOutput),
  _pointSpacing(np._pointSpacing),
  _segmentDescriptor(np._segmentDescriptor)
{
  strcpy(_inputFilename, np._inputFilename);
  memcpy(_neuronsPerLayer, np._neuronsPerLayer, 6*sizeof(int));
  _endNeurons = new int[_nSlicers];
  for (int i=0; i<_nSlicers; ++i) _endNeurons[i]=np._endNeurons[i];
  _neuronSegments = new int[_totalNeurons];
  for(int i=0; i<_totalNeurons; ++i) _neuronSegments[i]=np._neuronSegments[i];
}

NeuronPartitioner::~NeuronPartitioner() //Delete remaining data
{
  delete [] _neuronSegments;
  delete [] _endNeurons;
}

Decomposition* NeuronPartitioner::duplicate()
{
  return new NeuronPartitioner(*this);
}

void NeuronPartitioner::readFromFile(FILE* inputDataFile)
{
  size_t s=fread(_neuronsPerLayer, sizeof(int), 6, inputDataFile);
  s=fread(&_totalNeurons, sizeof(int), 1, inputDataFile);
#ifdef VERBOSE
  if (_rank == 0) std::cout<<"Read "<<_totalNeurons<<" neurons..."<<std::endl;
#endif  
  _neuronSegments = new int[_totalNeurons];
  for(int l = 0; l < _totalNeurons; l++) {
    s=fread(&_neuronSegments[l], sizeof(int), 1, inputDataFile);
    _totalSegmentsRead += _neuronSegments[l];
  }
}

void NeuronPartitioner::writeToFile(FILE* inputDataFile)
{
  fwrite(_neuronsPerLayer, sizeof(int), 6, inputDataFile);
  fwrite(&_totalNeurons, sizeof(int), 1, inputDataFile);
  for(int l = 0; l < _totalNeurons; l++) {
    fwrite(&_neuronSegments[l], sizeof(int), 1, inputDataFile);
  }
}

void NeuronPartitioner::partitionBinaryNeurons(int& nSlicers, const int nTouchDetectors, Tissue* tissue)
{ 
  _size = (nSlicers>nTouchDetectors)?nSlicers:nTouchDetectors;

  //Opens the files if it exist
  FILE* inputDataFile;
  
  if((inputDataFile = fopen(_inputFilename, "rb")) == 0) {
    printf("Could not find the input file %s...\n", _inputFilename);
    MPI_Finalize();
    exit(0);
  } 
  readFromFile(inputDataFile);
  if (_totalNeurons<nSlicers) nSlicers=_totalNeurons;
  _nSlicers = nSlicers;
  _endNeurons = new int[_nSlicers];

  decompose();

  tissue->setPartitioner(this);
  if (_rank<_nSlicers) tissue->loadBinary(inputDataFile,
					  _inputFilename, 
					  _neuronGroupNeuronCount,
					  _neuronGroupSegCount,
					  _neuronGroupStartNeuron,
					  this,
					  _resample,
					  _dumpOutput,
					  _pointSpacing);
  tissue->getTotalNumberOfBranches();

  fclose(inputDataFile);
}

void NeuronPartitioner::countAllNeurons(const char* inputFilename, int& totalNeurons, int& totalSegmentsRead, int* neuronsPerLayer, std::vector<int>& neuronSegments)
{
  totalNeurons=totalSegmentsRead=0;
  for (int i=0; i<6; ++i) neuronsPerLayer[i]=0;
  neuronSegments.clear();

  //Opens the files if it exist
  FILE *filenameFile, *inputDataFile;
    
  if((filenameFile = fopen(inputFilename, "rb")) == 0) {
    printf("Could not find the input file %s...\n", inputFilename);
    MPI_Finalize();
    exit(0);
  } 
  Params p;
  char bufS[1024], filename[256], offsetType, axonPar[256], basalPar[256], apicalPar[256];
  int layer, morphtype, electrotype, nsegs;
  int seg, branchType, parent;
  double x1, y1, z1, x2, y2, z2, r;
  p.skipHeader(filenameFile); 
  while (!feof(filenameFile)) {
    strcpy(bufS,"");
    char* c=fgets(bufS,1024,filenameFile);
    if (bufS[0]!='#' && bufS[0]!='\n') {
      if (11==sscanf(bufS, "%s %d %d %d %lf %lf %lf %c %s %s %s", filename, &layer, &morphtype, &electrotype, &x1, &y1, &z1, &offsetType, axonPar, basalPar, apicalPar)) {
	if((inputDataFile = fopen(filename, "rt")) == 0) {
	  printf("Could not find the input file %s...\n", filename);
	  MPI_Finalize();
	  exit(0);
	}
	int pos=ftell(inputDataFile);
	while (getc(inputDataFile)=='#') {
	  while (getc(inputDataFile)!='\n') {}
	  pos=ftell(inputDataFile);
	}
	fseek(inputDataFile, pos, SEEK_SET);
	assert (!feof(inputDataFile));
	nsegs=0;
	while (fscanf(inputDataFile, "%d %d %lf %lf %lf %lf %d", &seg, &branchType, &x2, &y2, &z2, &r, &parent)!=EOF) {
	  if (branchType!=1 || parent!=1) ++nsegs;
	}
	fclose(inputDataFile);

	neuronSegments.push_back(nsegs);
	totalSegmentsRead+=nsegs;
	++totalNeurons;
	++neuronsPerLayer[layer];
      }
    }
  }
  fclose(filenameFile);

}

void NeuronPartitioner::partitionTextNeurons(int& nSlicers, const int nTouchDetectors, Tissue* tissue)
{ 
  _size = (nSlicers>nTouchDetectors)?nSlicers:nTouchDetectors;
  _totalNeurons=_totalSegmentsRead=0;
  for (int i=0; i<6; ++i) _neuronsPerLayer[i]=0;
  std::vector<int> neuronSegments;

  if (_rank==0)
    countAllNeurons(_inputFilename, _totalNeurons, _totalSegmentsRead, _neuronsPerLayer, neuronSegments);

  MPI_Bcast(&_totalSegmentsRead, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&_totalNeurons, 1, MPI_INT, 0, MPI_COMM_WORLD);
  _neuronSegments = new int[_totalNeurons];
  if (_rank==0) for(int i=0; i<_totalNeurons; ++i) _neuronSegments[i]=neuronSegments[i];
  MPI_Bcast(_neuronsPerLayer, 6, MPI_INT, 0, MPI_COMM_WORLD); 
  MPI_Bcast(_neuronSegments, _totalNeurons, MPI_INT, 0, MPI_COMM_WORLD);
  
  if (_rank == 0) printf("\nTotal Neuron = %d\n\n",_totalNeurons);

  if (_totalNeurons<nSlicers) nSlicers=_totalNeurons;
  _nSlicers = nSlicers;
  _endNeurons = new int[_nSlicers];
  
  decompose();

  tissue->setPartitioner(this);
  if (_rank<_nSlicers) tissue->loadText( _inputFilename, 
					 _neuronGroupNeuronCount,
					 _neuronGroupSegCount, 
					 _neuronGroupStartNeuron, 
					 this, 
					 _resample,
					 _dumpOutput,
					 _pointSpacing);
  tissue->getTotalNumberOfBranches();
}

void NeuronPartitioner::decompose()
{  
  //decompose the neurons depending on their number of segments
  int currentNeuron=0;
  int startNeuron=0;
  _neuronGroupNeuronCount = 0;
  _neuronGroupSegCount = 0;
  _neuronGroupStartNeuron=0;
  int segsAlloc=0;
  double segsPerProc=0;
  
  for (int i=0; i<_nSlicers; ++i) {
    if (_totalNeurons-currentNeuron==_nSlicers-i) { 
      for (; i<_nSlicers; ++i) {
	startNeuron=_endNeurons[i]=currentNeuron;
	++currentNeuron;
	if (_rank==i) {
	  _neuronGroupStartNeuron = startNeuron;
	  _neuronGroupNeuronCount = 1;
	  _neuronGroupSegCount = _neuronSegments[startNeuron];
	}
      }
      break;
    }
    
    segsPerProc = double(_totalSegmentsRead-segsAlloc)/double(_nSlicers-i); 
    startNeuron=currentNeuron;
    int segsAllocHere=0;
    while (_totalNeurons-currentNeuron>_nSlicers-i-1) {
      segsAllocHere += _neuronSegments[currentNeuron];
      ++currentNeuron;
      if (currentNeuron>=_totalNeurons || 
	  segsPerProc-double(segsAllocHere)<0.5*double(_neuronSegments[currentNeuron])) {
	break;
      }
    }
    _endNeurons[i]=currentNeuron-1;
    if (_rank==i) {
      _neuronGroupStartNeuron = startNeuron;
      _neuronGroupNeuronCount = currentNeuron-startNeuron;
      _neuronGroupSegCount = segsAllocHere;
    }
    segsAlloc = segsAlloc + segsAllocHere;
  }
}

void NeuronPartitioner::getRanks(Sphere* sphere, double* coords2, double deltaRadius, ShallowArray<int, MAXRETURNRANKS, 100>& ranks)
{
  ranks.clear();
  ranks.push_back(getNeuronRank(_segmentDescriptor.getNeuronIndex(sphere->_key)));  
}

void NeuronPartitioner::addRanks(Sphere* sphere, double* coords2, double deltaRadius, ShallowArray<int, MAXRETURNRANKS, 100>& ranks)
{
  ranks.push_back(getNeuronRank(_segmentDescriptor.getNeuronIndex(sphere->_key)));  
}

bool NeuronPartitioner::mapsToRank(Sphere* sphere, double* coords2, double deltaRadius, int rank)
{
  return (rank==getNeuronRank(_segmentDescriptor.getNeuronIndex(sphere->_key)));
}

void NeuronPartitioner::getRanks(Sphere* sphere, double deltaRadius, ShallowArray<int, MAXRETURNRANKS, 100>& ranks)
{
  ranks.clear();
  ranks.push_back(getNeuronRank(_segmentDescriptor.getNeuronIndex(sphere->_key)));  
}

int NeuronPartitioner::getRank(Sphere& sphere)
{
  return getNeuronRank(_segmentDescriptor.getNeuronIndex(sphere._key));
}

int NeuronPartitioner::getNeuronRank(int neuronIndex)
{
  int rank=0;
  for (; rank<_nSlicers; ++rank) {
    if (neuronIndex<=_endNeurons[rank]) break;
  }
  return rank;
}

