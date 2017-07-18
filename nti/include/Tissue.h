// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// ================================================================

#ifndef TISSUE_H
#define TISSUE_H

#include <mpi.h>

#include <iostream>
#include <vector>
#include <math.h>
//#include <string.h>
#include <string>
#include <limits.h>

#include "Neuron.h"
#include "Communicator.h"
#include "NeuronPartitioner.h"
#include "SegmentSpace.h"
#include "SegmentDescriptor.h"

#ifdef BINARY64BITS
typedef long long PosType;
#define MPI_POS_TYPE MPI_LONG_LONG_INT
#else
// typedef long int PosType;
typedef int PosType;
//#define MPI_POS_TYPE MPI_LONG
#define MPI_POS_TYPE MPI_INT
#endif

#define NEURON_COUNT_SIZE 28
#define NEURON_SIZE 48
#define BRANCH_SIZE 8
#define SEGMENT_SIZE 40

class Neuron;
class Segment;
class SegmentSpace;
class Params;

class Tissue {
  public:
  Tissue(int size, int rank, bool logTranslationHistory = false,
         bool logRotationHistory = false);
  ~Tissue();

  void loadBinary(FILE*, const std::string&, const int, const int,
                  const int startNeuron, NeuronPartitioner*, bool resample,
                  bool dumpOutput, double pointSpacing);
  void loadText(const std::string&, const int, const int, const int,
                NeuronPartitioner*, bool resample, bool dumpOutput,
                double pointSpacing);
  void setPartitioner(NeuronPartitioner* neuronPartitioner) {
    _neuronPartitioner = neuronPartitioner;
  }
  void resampleNeurons(const int neuronArraySize,
                       std::vector<Segment>& segments, double pointSpacing);
  void resetBranchRoots(const int neuronArraySize,
                        std::vector<Segment>& segments);
  void updateCellBodies();

  void updateBranchRoots(int frontNumber);
  void updateFront(int frontNumber);
  void resetSegments(std::vector<Segment>& segments, bool resampled);
  void writeSegmentCounts(FILE*, PosType bin_start_pos);
  void writeCoordinates(FILE*, PosType bin_start_pos);
  void generateBins(double*&, int*&, double*&, double*&, double*&);
  void generateHistogram(int&, int**&, SegmentSpace* segmentSpace = 0,
                         TouchSpace* touchSpace = 0);
  void generateAlternateHistogram();
  void outputHistogram(FILE* outputDataFile);
  void openLogFiles();
  const int getNeuronArraySize() { return _neuronArraySize; }
  const int getNeuronIndex() { return _neuronIndex; }
  const int getSegmentArraySize() { return _segmentArraySize; }
  int getRank() { return _rank; }

  Neuron* getNeurons() { return _neurons; }
  Segment* getSegments() { return _segments; }
  NeuronPartitioner* getNeuronPartitioner() { return _neuronPartitioner; }

  int getNeuronIndex(int globalNeuronIndex);
  bool isInTissue(int neuronIndex);
  void rotateNeuronY(int neuronIndex, double rotation, int iteration);
  void translateNeuron(int neuronIndex, double translation[3], int iteration);
  int getMaxFrontNumber() { return _maxFrontNumber; }
  int getMaxBranchOrder() { return _maxBranchOrder; }
  int getTotalNumberOfBranches();
  int getTotalSegments() { return _totalSegments; }
  int getTotal() { return _total; }
  bool isEmpty() { return _isEmpty; }
  int outputTextNeurons(std::string outExtension, FILE* tissueOutFile,
                        int globalOffset);
  int outputTextNeuron(int neuronID, std::string outName, FILE* tissueOutFile,
                       int globalOffset);
  void outputBinaryNeurons(std::string outName);
  void writeForcesToFile();
  void getVisualizationSpheres(SegmentSpace&, int& nspheres, float*& positions,
                               float*& radii, int*& types);
  void clearSegmentForces();
  // void getHistogram(int**& histogram, double*& minXYZ, double*& maxXYZ,
  // double*& binwidth, int*& nbinsXYZ, int& total);
  void getLocalHistogram(int**& histogram, double*& minXYZ, double*& maxXYZ,
                         double*& binwidth, int*& nbinsXYZ);

  private:
  void setRootSegments();
  void setBranchOrders();
  void setUpSegmentIDs();

  int _neuronArraySize, _segmentArraySize, _neuronIndex;

  Neuron* _neurons;
  Segment* _segments;
  int _maxBranchOrder;
  NeuronPartitioner* _neuronPartitioner;
  int _size;
  int _rank;
  int _maxFrontNumber;
  int _totalBranches;
  int _totalSegments;
  int _total;

  bool _logTranslationHistory;
  bool _logRotationHistory;
  FILE* _neuronTranslationFile;
  FILE* _neuronRotationFile;

  bool _isEmpty;
  PosType _pos;
  double _maxXYZ[3], _minXYZ[3], _columnSizeXYZ[3], _binwidth[3];
  int _nbinsXYZ[3], _nbinsMaxXYZ[3], *_histogramXYZ[3], *_localHistogramXYZ[3];

  // char _inFilename[256];
  std::string _inFilename;
  int _nspheresAllocated;
  std::vector<std::string> _neuronOutputFilenames;
  SegmentDescriptor _segmentDescriptor;
};

#endif

