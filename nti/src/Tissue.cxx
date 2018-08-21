// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-07-18-2017
//
// (C) Copyright IBM Corp. and EPFL 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// ================================================================

#include "Tissue.h"
#include "NeuronPartitioner.h"
#include "Neuron.h"
#include "Segment.h"
#include "History.h"
#include "Branch.h"
#include "Segment.h"
#include "SegmentForce.h"
#include "SegmentSpace.h"
#include "TouchSpace.h"
#include "Params.h"
#include "AllInSegmentSpace.h"
#include "VecPrim.h"
#include <cassert>
#include <sstream>
#include <map>
#include <cstdlib>

#define FIRST_CAPSULE_CONSIDER_AT_SOMA_BORDER

#define SUGGESTED_BIN_WIDTH 1.0
#define MIN_COLUMN_DIM 0.000000000001
//#define COMPOSITE_OUTPUT
#define TISSUE_COORD_SWC_OUT

Tissue::Tissue(int size, int rank, bool logTranslationHistory,
               bool logRotationHistory)
    : _neuronArraySize(0),
      _segmentArraySize(0),
      _neuronIndex(0),
      _neurons(0),
      _segments(0),
      _maxBranchOrder(0),
      _neuronPartitioner(0),
      _size(size),
      _rank(rank),
      _maxFrontNumber(0),
      _totalBranches(-1),
      _totalSegments(0),
      _total(0),
      _logTranslationHistory(logTranslationHistory),
      _logRotationHistory(logRotationHistory),
      _neuronTranslationFile(0),
      _neuronRotationFile(0),
      _isEmpty(true),
      _pos(0),
      _nspheresAllocated(0) {
  for (int d = 0; d < 3; ++d) {
    _histogramXYZ[d] = 0;
    _localHistogramXYZ[d] = 0;
    _nbinsMaxXYZ[d] = 0;
  }
}

void Tissue::loadBinary(FILE* inputDataFile, const std::string& inputFilename,
                        const int neuronArraySize, const int segmentArraySize,
                        const int startNeuron,
                        NeuronPartitioner* neuronPartitioner, bool resample,
                        bool dumpOutput, double pointSpacing) {
  _neuronIndex = startNeuron;
  _neuronArraySize = neuronArraySize;
  _segmentArraySize = segmentArraySize;
  assert(_neuronPartitioner == neuronPartitioner);
  _isEmpty = false;

  assert(_neuronArraySize > 0);
  assert(_segmentArraySize > 0);
  _neurons = new Neuron[_neuronArraySize];     // Create the array of Neurons
  _segments = new Segment[_segmentArraySize];  // Create the array of segments
  Segment* segmentPtr = _segments;
  std::vector<Segment> segments;

  openLogFiles();

  // Finds the positions in the binary file

  // Finds the positions in the binary file
  PosType binaryPosition, pos;

#ifdef BINARY64BITS
  // reads the positions of the neurons
  fseeko64(inputDataFile, sizeof(PosType) * startNeuron, SEEK_CUR);
  _pos = ftello64(inputDataFile);

  for (int i = 0; i < neuronArraySize; i++) {
    fread(&binaryPosition, sizeof(PosType), 1, inputDataFile);
    pos = ftello64(inputDataFile);
    fseeko64(inputDataFile, binaryPosition, SEEK_SET);
    segmentPtr = _neurons[i].load(inputDataFile, segmentPtr, this, i,
                                  _neuronTranslationFile, _neuronRotationFile);
    fseeko64(inputDataFile, pos, SEEK_SET);
  }
#else
  // reads the positions of the neurons
  fseek(inputDataFile, sizeof(PosType) * startNeuron, SEEK_CUR);
  _pos = ftello64(inputDataFile);
  size_t s;
  for (int i = 0; i < neuronArraySize; i++) {
    s = fread(&binaryPosition, sizeof(PosType), 1, inputDataFile);
    pos = ftell(inputDataFile);
    fseek(inputDataFile, binaryPosition, SEEK_SET);
    segmentPtr =
        _neurons[i].loadBinary(inputDataFile, segmentPtr, this, i,
                               _neuronTranslationFile, _neuronRotationFile);
    fseek(inputDataFile, pos, SEEK_SET);
  }
#endif
  //  strcpy(_inFilename, inputFilename);
  _inFilename = inputFilename;
  if (resample)
    resampleNeurons(neuronArraySize, segments, pointSpacing);
  else
    resetBranchRoots(neuronArraySize, segments);
  setBranchOrders();
  setUpSegmentIDs();
}

void Tissue::loadText(const std::string& inputFilenames,
                      const int neuronArraySize, const int segmentArraySize,
                      const int startNeuron,
                      NeuronPartitioner* neuronPartitioner, bool resample,
                      bool dumpOutput, double pointSpacing) {
  _neuronIndex = startNeuron;
  _neuronArraySize = neuronArraySize;
  _segmentArraySize = segmentArraySize;
  assert(_neuronPartitioner == neuronPartitioner);
  _isEmpty = false;

  assert(_neuronArraySize > 0);
  assert(_segmentArraySize > 0);
  _neurons = new Neuron[_neuronArraySize];     // Create the array of Neurons
  _segments = new Segment[_segmentArraySize];  // Create the array of segments
  Segment* segmentPtr = _segments;
  std::vector<Segment> segments;

  openLogFiles();

  // Opens the files if it exist
  FILE* filenameFile, *inputDataFile;
  char filename[256], offsetType, axonPar[256], basalPar[256], apicalPar[256];
  int layer, morphtype, electrotype, nsegs;
  double x, y, z;

  if ((filenameFile = fopen(inputFilenames.c_str(), "rt")) == 0) {
    printf("Could not find the input filenames file %s...\n",
           inputFilenames.c_str());
    MPI_Finalize();
    exit(0);
  }

  Params p;

  p.skipHeader(filenameFile);
  std::map<std::string, int> filenameMap;
  std::map<std::string, int>::iterator filenameMapIter;
  char bufS[1024];
  char* c;
  for (int i = 0; i < startNeuron; ++i) {
    // strcpy(bufS, "");
    bufS[0] = '\n';
    do {
      c = fgets(bufS, 1024, filenameFile);
    } while (bufS[0] == '#' || bufS[0] == '\n');
    if (11 == sscanf(bufS, "%s %d %d %d %lf %lf %lf %c %s %s %s", filename,
                     &layer, &morphtype, &electrotype, &x, &y, &z, &offsetType,
                     axonPar, basalPar, apicalPar)) {
      std::string fname(filename);
      filenameMapIter = filenameMap.find(fname);
      if (filenameMapIter == filenameMap.end())
        filenameMap[fname] = 0;
      else
        ++filenameMapIter->second;
    }
  }

  for (int i = 0; i < _neuronArraySize; ++i) {
    // strcpy(bufS, "");
    bufS[0] = '\n';
    do {
      c = fgets(bufS, 1024, filenameFile);
    } while (bufS[0] == '#' || bufS[0] == '\n');
    if (11 == sscanf(bufS, "%s %d %d %d %lf %lf %lf %c %s %s %s", filename,
                     &layer, &morphtype, &electrotype, &x, &y, &z, &offsetType,
                     axonPar, basalPar, apicalPar)) {
      std::string fname(filename);
      filenameMapIter = filenameMap.find(fname);
      if (filenameMapIter == filenameMap.end())
        filenameMap[fname] = 0;
      else
        ++filenameMapIter->second;

      if ((inputDataFile = fopen(filename, "rt")) == 0) {
        printf("Could not find the input file %s...\n", filename);
        MPI_Finalize();
        exit(0);
      }
      std::ostringstream count;
      count << "_" << filenameMap[fname];
      fname.insert(fname.length() - 4, count.str());

      _neuronOutputFilenames.push_back(fname);
      p.skipHeader(inputDataFile);
      assert(!feof(inputDataFile));
      segmentPtr = _neurons[i].loadText(inputDataFile, segmentPtr, this, i,
                                        _neuronTranslationFile,
                                        _neuronRotationFile, layer, morphtype,
                                        electrotype, x, y, z, offsetType);
      fclose(inputDataFile);
    }
  }
  if (segmentPtr != _segments + _segmentArraySize) {
    std::cerr << segmentPtr - (_segments + _segmentArraySize) << std::endl;
    assert(0);
  }
  fclose(filenameFile);

  // strcpy(_inFilename, inputFilenames);
  _inFilename = inputFilenames;

  if (resample)
    resampleNeurons(neuronArraySize, segments, pointSpacing);
  else
    resetBranchRoots(neuronArraySize, segments);
  setBranchOrders();
  setUpSegmentIDs();
  if (resample && dumpOutput) {
    std::ostringstream os;
    os << ".resampled." << pointSpacing;
    outputTextNeurons(os.str(), 0, 0);
  }
}

void Tissue::resampleNeurons(const int neuronArraySize,
                             std::vector<Segment>& segments,
                             double pointSpacing) {
  assert(!isEmpty());
  for (int i = 0; i < neuronArraySize; ++i)
    _neurons[i].resample(segments, pointSpacing);
  resetSegments(segments, true);
  for (int i = 0; i < neuronArraySize; ++i) _neurons[i].eliminateLostBranches();
}

void Tissue::resetBranchRoots(const int neuronArraySize,
                              std::vector<Segment>& segments) {
  assert(!isEmpty());
  for (int i = 0; i < neuronArraySize; ++i)
    _neurons[i].resetBranchRoots(segments);
  resetSegments(segments, false);
  setRootSegments();
}

void Tissue::updateCellBodies() {
  for (int i = 0; i < _segmentArraySize; ++i) {
    if (_segments[i].getBranch()->getBranchOrder() == 0 &&
        _segments[i].getSegmentIndex() == 1) {
      std::copy(_segments[i].getCoords(), _segments[i].getCoords() + 3,
                _segments[i - 1].getCoords());
      //            memcpy(_segments[i - 1].getCoords(),
      // _segments[i].getCoords(), sizeof(double) * 3);
    }
  }
}

void Tissue::updateBranchRoots(int frontNumber) {
  if (frontNumber != 0) {
    for (int i = 0; i < _segmentArraySize; ++i) {
      if (frontNumber == _segments[i].getFrontLevel() &&
          _segments[i].getSegmentIndex() == 0) {
        double* coords = _segments[i].getCoords();
        double* velocity = _segments[i].getVelocity();
        assert(_segments[i].getFrontLevel() ==
               _segments[i].getBranch()->getRootSegment()->getFrontLevel());
        Segment* root = _segments[i].getBranch()->getRootSegment();
        double* rootCoords = root->getCoords();
        double* rootVelocity = root->getVelocity();
        std::copy(rootCoords, rootCoords + 3, coords);
        // memcpy(coords, rootCoords, sizeof(double) * 3);
        std::copy(rootVelocity, rootVelocity + 3, velocity);
        // memcpy(velocity, rootVelocity, sizeof(double) * 3);
        std::fill(rootVelocity, rootVelocity + 3, 0);
        // memset(rootVelocity, 0, sizeof(double) * 3);
      }
    }
  }
}

void Tissue::updateFront(int frontNumber) {
  for (int i = 0; i < _segmentArraySize; ++i) {
    if (frontNumber == _segments[i].getFrontLevel() &&
        _segments[i].getSegmentIndex() > 0) {
      double* coords = _segments[i].getCoords();
      double* prevCoords = _segments[i - 1].getCoords();
      double* origCoords = _segments[i].getOrigCoords();
      double* prevOrigCoords = _segments[i - 1].getOrigCoords();
      double* velocity = _segments[i].getVelocity();
      double* prevVelocity = _segments[i - 1].getVelocity();
      for (int ii = 0; ii < 3; ++ii) {
        coords[ii] = prevCoords[ii] + (origCoords[ii] - prevOrigCoords[ii]);
        velocity[ii] = prevVelocity[ii];
        prevVelocity[ii] = 0;
      }
    }
  }
}
void Tissue::resetSegments(std::vector<Segment>& segments, bool resampled) {
  Segment* oldSegments = _segments;
  _segmentArraySize = segments.size();
  _segments =
      new Segment[_segmentArraySize];  // Create the new array of segments
  Branch* thisBranch = 0, * lastBranch = 0;
  Neuron* thisNeuron = 0, * lastNeuron = 0;
  for (int i = 0; i < _segmentArraySize; ++i) {
    segments[i].setKey();
    _segments[i] = segments[i];
    thisBranch = _segments[i].getBranch();
    if (thisBranch != lastBranch) {
      if (resampled && thisBranch->getBranchIndex() != 0) {
        thisBranch->resetSegments(
            &_segments[i], &_segments[thisBranch->getRootSegment()
                                          ->getBranch()
                                          ->getResampledTerminalIndex()]);
      } else
        thisBranch->resetSegments(&_segments[i]);
      lastBranch = thisBranch;
    }
    thisNeuron = thisBranch->getNeuron();
    if (thisNeuron != lastNeuron) {
      if (lastNeuron) {
        lastNeuron->setSegmentsEnd(&_segments[i]);
      }
      thisNeuron->setSegmentsBegin(&_segments[i]);
      lastNeuron = thisNeuron;
    }
  }
  if (thisNeuron) {
    thisNeuron->setSegmentsEnd(_segments + _segmentArraySize);
  }
  delete[] oldSegments;
}

void Tissue::writeSegmentCounts(FILE* outputDataFile, PosType bin_start_pos) {
#ifdef BINARY64BITS
  fseeko64(outputDataFile,
           bin_start_pos + NEURON_COUNT_SIZE + sizeof(int) * _neuronIndex,
           SEEK_SET);
#else
  fseek(outputDataFile,
        bin_start_pos + NEURON_COUNT_SIZE + sizeof(int) * _neuronIndex,
        SEEK_SET);
#endif
  for (int i = 0; i < _neuronArraySize; ++i) {
    int nsegs = _neurons[i].getNumberOfSegments();
    fwrite(&nsegs, sizeof(int), 1, outputDataFile);
  }
}

void Tissue::writeCoordinates(FILE* outputDataFile, PosType bin_start_pos) {
  assert(_neuronPartitioner);
  long int rval = 0;
  for (int i = 0; i < _neuronArraySize; ++i) {
#ifdef BINARY64BITS
    PosType pos = ftello64(outputDataFile);
    fseek(outputDataFile,
          bin_start_pos + NEURON_COUNT_SIZE +
              sizeof(int) * _neuronPartitioner->getTotalNeurons() +
              sizeof(PosType) * (_neuronIndex + i),
          SEEK_SET);
    fwrite(&pos, sizeof(PosType), 1, outputDataFile);
    fseeko64(outputDataFile, pos, SEEK_SET);
#else
    PosType pos = ftell(outputDataFile);
    fseek(outputDataFile,
          bin_start_pos + NEURON_COUNT_SIZE +
              sizeof(int) * _neuronPartitioner->getTotalNeurons() +
              sizeof(PosType) * (_neuronIndex + i),
          SEEK_SET);
    fwrite(&pos, sizeof(PosType), 1, outputDataFile);
    fseek(outputDataFile, pos, SEEK_SET);
#endif
    _neurons[i].writeCoordinates(outputDataFile);
  }
}

void Tissue::generateBins(double*& columnSizeXYZ, int*& nbinsXYZ,
                          double*& binwidth, double*& maxXYZ, double*& minXYZ) {
  double localMaxXYZ[3] = {INT_MIN, INT_MIN, INT_MIN},
         localMinXYZ[3] = {INT_MAX, INT_MAX, INT_MAX};
  for (int i = 0; i < _segmentArraySize; ++i) {
    for (int d = 0; d < 3; ++d) {
      if (_segments[i].getCoords()[d] > localMaxXYZ[d])
        localMaxXYZ[d] = _segments[i].getCoords()[d];
      if (_segments[i].getCoords()[d] < localMinXYZ[d])
        localMinXYZ[d] = _segments[i].getCoords()[d];
    }
  }
  MPI_Allreduce((void*)localMaxXYZ, (void*)_maxXYZ, 3, MPI_DOUBLE, MPI_MAX,
                MPI_COMM_WORLD);
  MPI_Allreduce((void*)localMinXYZ, (void*)_minXYZ, 3, MPI_DOUBLE, MPI_MIN,
                MPI_COMM_WORLD);
  for (int d = 0; d < 3; ++d) {
    _columnSizeXYZ[d] = (_maxXYZ[d] - _minXYZ[d] == 0)
                            ? MIN_COLUMN_DIM
                            : _maxXYZ[d] - _minXYZ[d];
    _nbinsXYZ[d] = int(ceil(_columnSizeXYZ[d] / SUGGESTED_BIN_WIDTH));
    assert(_nbinsXYZ[d] > 0);
    if (_nbinsXYZ[d] > _nbinsMaxXYZ[d]) {
      delete[] _histogramXYZ[d];
      delete[] _localHistogramXYZ[d];
      _nbinsMaxXYZ[d] = int(double(_nbinsXYZ[d]) * 1.1);
      _histogramXYZ[d] = new int[_nbinsMaxXYZ[d]];
      _localHistogramXYZ[d] = new int[_nbinsMaxXYZ[d]];
    }
    _binwidth[d] = _columnSizeXYZ[d] / double(_nbinsXYZ[d]);
  }
  columnSizeXYZ = _columnSizeXYZ;
  nbinsXYZ = _nbinsXYZ;
  binwidth = _binwidth;
  maxXYZ = _maxXYZ;
  minXYZ = _minXYZ;
}

void Tissue::generateHistogram(int& total, int**& histogramXYZ,
                               SegmentSpace* segmentSpace,
                               TouchSpace* touchSpace) {
  assert(segmentSpace == 0 || touchSpace == 0);
  int localTotal = 0;
  for (int d = 0; d < 3; ++d) {
    for (int i = 0; i < _nbinsXYZ[d]; ++i) {
      _localHistogramXYZ[d][i] = 0;
      for (int j = 0; j < _segmentArraySize; ++j) {
        if ((segmentSpace ? segmentSpace->isInSpace(&_segments[j]) : true) &&
            (touchSpace ? touchSpace->isInSpace(_segments[j].getSegmentKey())
                        : true) &&
            _segments[j].getCoords()[d] >= _minXYZ[d] + (i * _binwidth[d]) &&
            (i == _nbinsXYZ[d] - 1 ||
             _segments[j].getCoords()[d] <
                 _minXYZ[d] + ((i + 1) * _binwidth[d]))) {
          ++_localHistogramXYZ[d][i];
          if (d == 0) ++localTotal;
        }
      }
    }
    MPI_Allreduce((void*)_localHistogramXYZ[d], (void*)_histogramXYZ[d],
                  _nbinsXYZ[d], MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  }
  MPI_Allreduce(&localTotal, &_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  _totalSegments = total = _total;
  histogramXYZ = _histogramXYZ;
}

void Tissue::generateAlternateHistogram() {
  int total[3] = {0, 0, 0};
  for (int d = 0; d < 3; ++d) {
    for (int i = 0; i < _nbinsXYZ[d]; ++i) {
      total[d] += _histogramXYZ[d][i];
      _histogramXYZ[d][i] = 0;
    }
    MPI_Allreduce((void*)_localHistogramXYZ[d], (void*)_histogramXYZ[d],
                  _nbinsXYZ[d], MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    double denominator = 0;
    for (int i = 0; i < _nbinsXYZ[d]; ++i)
      denominator += double(_histogramXYZ[d][i]);
    double numerator = double(total[d]);
    int t = 0;
    for (int i = 0; i < _nbinsXYZ[d] - 1; ++i) {
      double bin = double(_histogramXYZ[d][i]);
      t += (_histogramXYZ[d][i] = int(round(bin * numerator / denominator)));
    }
    t += (_histogramXYZ[d][_nbinsXYZ[d] - 1] = total[d] - t);
    assert(t == total[d]);
  }
}

void Tissue::outputHistogram(FILE* outputDataFile) {
  PosType histBytes =
      sizeof(PosType) +
      sizeof(int) * (4 + _nbinsXYZ[0] + _nbinsXYZ[1] + _nbinsXYZ[2]) +
      sizeof(double) * 9;
  fwrite(&histBytes, sizeof(PosType), 1, outputDataFile);
  fwrite(&_total, sizeof(int), 1, outputDataFile);
  fwrite(_columnSizeXYZ, sizeof(double), 3, outputDataFile);
  fwrite(_nbinsXYZ, sizeof(int), 3, outputDataFile);
  for (int d = 0; d < 3; ++d) {
    fwrite(_histogramXYZ[d], sizeof(int), _nbinsXYZ[d], outputDataFile);
  }
  fwrite(&_maxXYZ[0], sizeof(double), 1, outputDataFile);
  fwrite(&_minXYZ[0], sizeof(double), 1, outputDataFile);
  fwrite(&_maxXYZ[1], sizeof(double), 1, outputDataFile);
  fwrite(&_minXYZ[1], sizeof(double), 1, outputDataFile);
  fwrite(&_maxXYZ[2], sizeof(double), 1, outputDataFile);
  fwrite(&_minXYZ[2], sizeof(double), 1, outputDataFile);
}

void Tissue::openLogFiles() {
  if (_logTranslationHistory) {
    char tfname[256];
    sprintf(tfname, "outNeurons_%d-%d_Translations.bin", _neuronIndex,
            _neuronIndex + _neuronArraySize - 1);
    if ((_neuronTranslationFile = fopen(tfname, "wb")) == 0) {
      printf("Could not open the translation history output file %s!\n",
             tfname);
      MPI_Finalize();
      exit(0);
    }
  }

  if (_logRotationHistory) {
    char rfname[256];
    sprintf(rfname, "outNeurons_%d-%d_Rotations.bin", _neuronIndex,
            _neuronIndex + _neuronArraySize - 1);
    if ((_neuronRotationFile = fopen(rfname, "wb")) == 0) {
      printf("Could not open the rotation history output file %s!\n", rfname);
      MPI_Finalize();
      exit(0);
    }
  }
}

void Tissue::setRootSegments() {
  for (int i = 0; i < _neuronArraySize; ++i) _neurons[i].setRootSegments();
}

void Tissue::setBranchOrders() {
  for (int i = 0; i < _neuronArraySize; ++i) {
    int nBranches = _neurons[i].getNumberOfBranches();
    Branch* branches = _neurons[i].getBranches();
    branches[0].setBranchOrder(0);
    branches[0].setDist2Soma(0.0);
    int j = 1;
    for (int j = 1; j < nBranches; ++j) {
      Branch* proximalBranch = branches[j].getRootSegment()->getBranch();
      int branchOrder = 1;
      double dist2Soma = 0;
      while (proximalBranch->getBranchOrder() != 0) {
        ++branchOrder;
        dist2Soma += proximalBranch->getLength();
        proximalBranch = proximalBranch->getRootSegment()->getBranch();
        assert(proximalBranch);
        assert(branchOrder < 1000);
      }
#ifdef FIRST_CAPSULE_CONSIDER_AT_SOMA_BORDER
      //here no matter what physical location of the first capsule, we always treat its distance to soma is soma-radius
      if (proximalBranch->getBranchType() == Branch::_SOMA)
      {
        dist2Soma += proximalBranch->getSegments()[0].getRadius();  
      }
      else{
        dist2Soma += proximalBranch->getLength();
      }
#else
      dist2Soma += proximalBranch->getLength();
#endif
      branches[j].setBranchOrder(branchOrder);
      branches[j].setDist2Soma(dist2Soma);
      if (branchOrder > _maxBranchOrder) _maxBranchOrder = branchOrder;
    }
  }
}

void Tissue::setUpSegmentIDs() {
  for (int i = 0; i < _segmentArraySize; ++i) {
    _segments[i].setSegmentArrayIndex(i);

    int frontnum = _segments[i].getSegmentIndex();
    Branch* cbranch = _segments[i].getBranch();
    int brnch_id = cbranch->getBranchIndex();
    while (brnch_id > 0) {
      Segment* seg = cbranch->getRootSegment();
      frontnum += seg->getSegmentIndex();
      cbranch = seg->getBranch();
      brnch_id = cbranch->getBranchIndex();
    }

    _segments[i].setFrontLevel(frontnum);
    if (frontnum > _maxFrontNumber) _maxFrontNumber = frontnum;
    _segments[i].setKey();
  }
}

int Tissue::getNeuronIndex(int globalNeuronIndex) {
  return globalNeuronIndex - _neuronIndex;
}

bool Tissue::isInTissue(int neuronIndex) {
  return (neuronIndex >= 0 && neuronIndex < _neuronArraySize);
}

void Tissue::rotateNeuronY(int neuronIndex, double rotation, int iteration) {
  _neurons[neuronIndex].rotateY(rotation, iteration);
}

void Tissue::translateNeuron(int neuronIndex, double translation[3],
                             int iteration) {
  _neurons[neuronIndex].translate(translation, iteration);
}

int Tissue::outputTextNeurons(std::string outExtension, FILE* tissueOutFile,
                              int globalOffset) {
  int segmentsWritten = 0;
  for (int neuronID = 0; neuronID < _neuronArraySize; ++neuronID) {
    std::string outName = _neuronOutputFilenames[neuronID];
    outName.insert(outName.find_last_of('.'), outExtension);
    segmentsWritten += outputTextNeuron(neuronID, outName, tissueOutFile,
                                        globalOffset + segmentsWritten);
  }
  return segmentsWritten;
}

int Tissue::outputTextNeuron(int neuronID, std::string outName,
                             FILE* tissueOutFile, int globalOffset) {
  FILE* outfile = 0;
  int segCount = 0;
  Neuron* nPrev = 0;
  for (int i = 0; i < _segmentArraySize; ++i) {
    Segment* s = &_segments[i];
    Branch* b = s->getBranch();
    Neuron* n = b->getNeuron();
    if (n->getNeuronIndex() == neuronID) {
      if (outfile == 0) {
        if ((outfile = fopen(outName.c_str(), "wt")) == NULL) {
          printf("Could not open the output file %s!\n", outName.c_str());
          MPI_Finalize();
          exit(0);
        }
      }
      int neuronBegin = n->getSegmentsBegin()->getSegmentArrayIndex();
      int neuronEnd = neuronBegin + n->getNumberOfSegments();
      fprintf(outfile, "# output neuron %d, %d, %d, %d\n", n->getNeuronIndex(),
              i, neuronBegin, neuronEnd);
      for (; i < neuronEnd; ++i) {
        s = &_segments[i];
        b = s->getBranch();
        double* coords = s->getCoords();
        double radius = s->getRadius();
        int segmentIndex = i - neuronBegin - b->getBranchIndex();
        int rootSegmentIndex = -1;
        if (b->getSegments() < s) {
          if (b->getSegments() == s - 1) {
            Segment* root = b->getRootSegment();
            if (root)
              rootSegmentIndex = root->getSegmentArrayIndex() - neuronBegin -
                                 root->getBranch()->getBranchIndex();
          } else
            rootSegmentIndex = segmentIndex - 1;
          fprintf(outfile, "%d %d %lf %lf %lf %lf %d\n", segmentIndex,
                  b->getBranchType() + 1, coords[0], coords[1], coords[2],
                  radius, rootSegmentIndex);
          if (tissueOutFile) {
#ifdef COMPOSITE_OUTPUT
            fprintf(tissueOutFile, "%d %d %lf %lf %lf %lf %d\n",
                    ++segCount + globalOffset,
                    (b->getBranchType() == Branch::_SOMA &&
                     n->getGlobalNeuronIndex() > 0)
                        ? 2
                        : b->getBranchType() + 1,
                    coords[0], coords[1], coords[2], radius,
                    (rootSegmentIndex == -1) ? -1
                                             : rootSegmentIndex + globalOffset);
#else
            if (n != nPrev) {
              fprintf(tissueOutFile,
                      "%s %d %d %d %lf %lf %lf %c NULL NULL NULL\n",
                      outName.c_str(), n->getLayer(), n->getMorphologicalType(),
                      n->getElectrophysiologicalType(),
#ifdef TISSUE_COORD_SWC_OUT
                      0.0, 0.0, 0.0, 'A');
#else
                      n->getCenter()[0], n->getCenter()[1], n->getCenter()[2],
                      'A');
#endif
              nPrev = n;
            }
#endif
          }
        }
      }
    }
  }
  if (outfile) fclose(outfile);
  return segCount;
}

void Tissue::outputBinaryNeurons(std::string outName) {
  assert(_neuronPartitioner);
  AllInSegmentSpace s;
  FILE* inputDataFile;
  FILE* outputDataFile;
  int written = 0;
  int nextToWrite = 0;
  char infname[256];
  char outfname[256];

  if (outName == "NONE") {
    PosType dataPos = 0, pos = 0, nextPos = 0, sz, kb, rm;
    char kbyte[1000];
    std::fill_n(kbyte, 1000, 0);
    // memset(kbyte, 0, 1000);
    char byte;
    std::fill_n(&byte, 1, 0);
    // memset(&byte, 0, 1);

    while (nextToWrite < _size) {
      if (nextToWrite == _rank) {
        if (_rank < _size - 1) {
          sprintf(outfname, "tmp%d.bin", _rank % 2);
          if ((outputDataFile = fopen(outfname, "w+b")) == 0)
            printf("Could not open the output file %s, rank %d.\n", outfname,
                   _rank);
        } else {
          sprintf(outfname, "%s", outName.c_str());
          // strcpy(outfname, outName.c_str());
          if ((outputDataFile = fopen(outfname, "w+b")) == 0)
            printf("Could not open the output file %s, rank %d.\n", outfname,
                   _rank);
        }
        if (_rank > 0) {
          pos = nextPos;
          sprintf(infname, "tmp%d.bin", (_rank - 1) % 2);
          if ((inputDataFile = fopen(infname, "rb")) == 0) {
            printf("Could not open the input file %s, rank %d.\n", infname,
                   _rank);
          }
          size_t s = fread(&dataPos, sizeof(PosType), 1, inputDataFile);
#ifdef BINARY64BITS
          fseeko64(inputDataFile, 0, SEEK_END);
          sz = ftello64(inputDataFile);
#else
          fseek(inputDataFile, 0, SEEK_END);
          sz = ftell(inputDataFile);
#endif
          rewind(inputDataFile);
          kb = sz / 1000;
          rm = sz % 1000;
          for (PosType i = 0; i < kb; ++i) {
            s = fread(kbyte, 1000, 1, inputDataFile);
            fwrite(kbyte, 1000, 1, outputDataFile);
          }
          for (PosType i = 0; i < rm; ++i) {
            s = fread(&byte, 1, 1, inputDataFile);
            fwrite(&byte, 1, 1, outputDataFile);
          }
          fclose(inputDataFile);
        } else {
          outputHistogram(outputDataFile);
#ifdef BINARY64BITS
          dataPos = ftello64(outputDataFile);
#else
          dataPos = ftell(outputDataFile);
#endif
          int totalNeurons = _neuronPartitioner->getTotalNeurons();
          fwrite(_neuronPartitioner->getNeuronsPerLayer(), sizeof(int), 6,
                 outputDataFile);
          fwrite(&totalNeurons, sizeof(int), 1, outputDataFile);

          sz = totalNeurons * sizeof(int);
          kb = sz / 1000;
          rm = sz % 1000;
          for (PosType i = 0; i < kb; ++i)
            fwrite(kbyte, 1000, 1,
                   outputDataFile);  // placeholder for number of segments
          for (PosType i = 0; i < rm; ++i) fwrite(&byte, 1, 1, outputDataFile);

          sz = totalNeurons * sizeof(PosType);
          kb = sz / 1000;
          rm = sz % 1000;
          for (PosType i = 0; i < kb; ++i)
            fwrite(kbyte, 1000, 1, outputDataFile);  // placeholder for position
                                                     // of segments in file
          for (PosType i = 0; i < rm; ++i) fwrite(&byte, 1, 1, outputDataFile);

#ifdef BINARY64BITS
          pos = ftello64(outputDataFile);
#else
          pos = ftell(outputDataFile);
#endif
          sz = totalNeurons * NEURON_SIZE +
               getTotalNumberOfBranches() * BRANCH_SIZE +
               _neuronPartitioner->getTotalSegmentsRead() * SEGMENT_SIZE;
          kb = sz / 1000;
          rm = sz % 1000;
          for (PosType i = 0; i < kb; ++i)
            fwrite(kbyte, 1000, 1, outputDataFile);
          for (PosType i = 0; i < rm; ++i) fwrite(&byte, 1, 1, outputDataFile);
        }

        writeSegmentCounts(outputDataFile, dataPos);

#ifdef BINARY64BITS
        fseeko64(outputDataFile, pos, SEEK_SET);
#else
        fseek(outputDataFile, pos, SEEK_SET);
#endif
        writeCoordinates(outputDataFile, dataPos);

#ifdef BINARY64BITS
        pos = ftello64(outputDataFile);
#else
        pos = ftell(outputDataFile);
#endif
        fclose(outputDataFile);
        written = 1;
      }
      MPI_Allreduce((void*)&written, (void*)&nextToWrite, 1, MPI_INT, MPI_SUM,
                    MPI_COMM_WORLD);
      MPI_Allreduce((void*)&pos, (void*)&nextPos, 1, MPI_POS_TYPE, MPI_MAX,
                    MPI_COMM_WORLD);
    }
  }
}

int Tissue::getTotalNumberOfBranches() {
  if (_totalBranches == -1) {
    int localTotal = 0;
    for (int i = 0; i < _neuronArraySize; ++i) {
      localTotal += _neurons[i].getNumberOfBranches();
    }
    MPI_Allreduce((void*)&localTotal, (void*)&_totalBranches, 1, MPI_INT,
                  MPI_SUM, MPI_COMM_WORLD);
  }
  return _totalBranches;
}

void Tissue::writeForcesToFile() {
  FILE* data;
  PosType binpos = 0;
  char filename[256];
  sprintf(filename, "outForces_Tissue.%d", _rank);
  if ((data = fopen(filename, "wb")) == NULL) {
    printf("Could not open the output file %s!\n", filename);
    MPI_Finalize();
    exit(0);
  }
  int t = 1;
  fwrite(&t, sizeof(int), 1, data);
#ifdef BINARY64BITS
  binpos = ftello64(data);
#else
  binpos = ftell(data);
#endif
  int countForces = 0;
  fwrite(&countForces, sizeof(int), 1, data);

  Segment* segmentEnd = _segments + _segmentArraySize - 1, * s = _segments;
  int forceInfo[3];
  double s1Key;
  for (; s != segmentEnd; ++s, ++countForces) {
    double* f = s->getForce();
    forceInfo[0] = _segmentDescriptor.getNeuronIndex(f[0]);
    forceInfo[1] = _segmentDescriptor.getNeuronIndex(f[0]);
    forceInfo[2] = _segmentDescriptor.getBranchIndex(f[0]);
    fwrite(&forceInfo, sizeof(int), 3, data);
    fwrite(&f, 8, 3, data);
  }
#ifdef BINARY64BITS
  fseeko64(data, binpos, SEEK_SET);
#else
  fseek(data, binpos, SEEK_SET);
#endif
  fwrite(&countForces, sizeof(int), 1, data);
  fclose(data);
}

void Tissue::getVisualizationSpheres(SegmentSpace& space, int& nspheres,
                                     float*& positions, float*& radii,
                                     int*& types) {
  nspheres = 0;
  for (int i = 0; i < _segmentArraySize; ++i) {
    if (space.isInSpace(&_segments[i])) {
      ++nspheres;
      if (nspheres > _nspheresAllocated) {
        int n = nspheres;
        for (int j = i + 1; j < _segmentArraySize; ++j) {
          if (space.isInSpace(&_segments[j])) ++n;
        }
        n = int(double(n) * 1.1);
        float* pos = new float[n * 3];
        float* rad = new float[n];
        int* typ = new int[n];
        std::copy(positions, positions + 3 * _nspheresAllocated, pos);
        // memcpy(pos, positions, sizeof(float) * 3 * _nspheresAllocated);
        std::copy(radii, radii + _nspheresAllocated, rad);
        // memcpy(rad, radii, sizeof(float) * _nspheresAllocated);
        std::copy(types, types + _nspheresAllocated, typ);
        // memcpy(typ, types, sizeof(int) * _nspheresAllocated);
        delete[] positions;
        delete[] radii;
        delete[] types;
        _nspheresAllocated = n;
        positions = pos;
        radii = rad;
        types = typ;
      }

      int idx = nspheres - 1;
      radii[idx] = float(_segments[i].getRadius());
      types[idx] = _segmentDescriptor.getValue(
                       _segmentDescriptor.getSegmentKeyData("USER_FIELD_0"),
                       _segments[i].getSegmentKey()) &
                   0x7;
      idx *= 3;
      for (int j = 0; j < 3; ++j) {
        positions[idx + j] = float(_segments[i].getCoords()[j]);
      }
    }
  }
}

void Tissue::clearSegmentForces() {
  for (int i = 0; i < _segmentArraySize; ++i) {
    _segments[i].clearForce();
  }
}

void Tissue::getLocalHistogram(int**& histogram, double*& minXYZ,
                               double*& maxXYZ, double*& binwidth,
                               int*& nbinsXYZ) {
  histogram = _localHistogramXYZ;
  minXYZ = _minXYZ;
  maxXYZ = _maxXYZ;
  binwidth = _binwidth;
  nbinsXYZ = _nbinsXYZ;
}

Tissue::~Tissue() {
  delete[] _neurons;
  delete[] _segments;
  if (_logTranslationHistory) fclose(_neuronTranslationFile);
  if (_logRotationHistory) fclose(_neuronRotationFile);
  for (int d = 0; d < 3; ++d) {
    delete[] _localHistogramXYZ[d];
  }
}
