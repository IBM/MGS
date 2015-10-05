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

#ifndef NEURONPARTITIONER_H
#define NEURONPARTITIONER_H

#include <mpi.h>
#include "Decomposition.h"
#include "SegmentDescriptor.h"
#include "ShallowArray.h"

#include <iostream>
#include <stdlib.h>
#include <string>

class SegmentSpace;
class TouchSpace;

class Tissue;

class NeuronPartitioner : public Decomposition {
   public:
  NeuronPartitioner(int rank, const std::string& inputFilename, bool resample, bool dumpOutput,double pointSpacing);
  NeuronPartitioner(NeuronPartitioner&);
  virtual ~NeuronPartitioner();
  Decomposition* duplicate();

  void readFromFile(FILE*);
  void writeToFile(FILE*);
  static void countAllNeurons(const std::string& inputFilename, int& totalNeurons, int& totalSegmentsRead, int* neuronsPerLayer, std::vector<int>& neuronSegments);
  void partitionBinaryNeurons(int& nSlicers, const int nTouchDetectors, Tissue* tissue);
  void partitionTextNeurons(int& nSlicers, const int nTouchDetectors, Tissue* tissue);
  int getNumberOfSlicers() {return _nSlicers;}
  int* getNeuronsPerLayer() {return _neuronsPerLayer;}
  int getTotalNeurons() {return _totalNeurons;}
  int getTotalSegmentsRead() {return _totalSegmentsRead;}

  void decompose();
  void getRanks(Sphere* sphere, double* coords2, double deltaRadius, ShallowArray<int, MAXRETURNRANKS, 100>& ranks);
  void addRanks(Sphere* sphere, double* coords2, double deltaRadius, ShallowArray<int, MAXRETURNRANKS, 100>& ranks);
  bool mapsToRank(Sphere* sphere, double* coords2, double deltaRadius, int rank);
  void resetCriteria(SegmentSpace* segmentSpace) {assert(0);}
  void resetCriteria(TouchSpace* touchSpace) {assert(0);}

  void getRanks(Sphere* sphere, double deltaRadius, ShallowArray<int, MAXRETURNRANKS, 100>& ranks);
  int getRank(Sphere& sphere);
  bool isCoordinatesBased() {return false;}
  int getRank() {return _rank;}
  int getNeuronRank(int neuronIndex);

 private:

  //TUAN: potential error if length > 256
  // char _inputFilename[256]; //input filename  
  std::string _inputFilename;
  int _nSlicers;
  int _size;//number of cpus
  int _rank;//current cpu
      
  int _neuronsPerLayer[6];
  int _totalNeurons;
  int _totalSegmentsRead; // note: the number of segments changes after resampling
  bool _logTranslationHistory;
  bool _logRotationHistory;
  int* _endNeurons;
  int _neuronGroupNeuronCount;
  int _neuronGroupSegCount;
  int _neuronGroupStartNeuron;
  int* _neuronSegments;
  bool _resample;
  bool _dumpOutput;
  double _pointSpacing;
  SegmentDescriptor _segmentDescriptor;
};

#endif
