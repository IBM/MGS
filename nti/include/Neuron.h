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

#ifndef NEURON_H
#define NEURON_H

#include <mpi.h>

#include <iostream>
#include <math.h>
#include <vector>

#define DEG_RAD 0.017453292f
#define RAD_DEG 57.29577951f

#include "Segment.h"

class Branch;
class Tissue;
class History;

class Neuron {
  public:
  Neuron();

  Segment* loadBinary(FILE*, Segment*, Tissue*, const int neuronIndex, FILE*,
                      FILE*);
  Segment* loadText(FILE*, Segment*, Tissue*, const int neuronIndex, FILE*,
                    FILE*, int layer, int morphologicalType,
                    int electrophysiologicalType, double x, double y, double z,
                    char offsetType);
  void resample(std::vector<Segment>&, double pointSpacing);
  void resetBranchRoots(std::vector<Segment>&);
  void eliminateLostBranches();
  void writeCoordinates(FILE*);
  double* getCenter(void) { return _center; }
  const int getLayer(void) { return _layer; }
  const int getNumberOfBranches() { return _numberOfBranches; }
  const int getNumberOfSegments() { return _segmentsEnd - _segmentsBegin; }
  const int getMorphologicalType() { return _morphologicalType; }
  const int getElectrophysiologicalType() { return _electrophysiologicalType; }
  const int getNeuronIndex(void) { return _neuronIndex; }
  const int getGlobalNeuronIndex() { return _globalNeuronIndex; }

  void translate(double xyz[3], int iteration);
  void rotateY(double angle, int iteration);
  History* getTranslationHistory() { return _translationHistory; }
  History* getRotationYHistory() { return _rotationYHistory; }

  Branch* getBranches() { return _branch; }
  Tissue* getNeuronGroup() { return _neuronGroup; }
  Segment* getSegmentsBegin() { return _segmentsBegin; }
  Segment* getSegmentsEnd() { return _segmentsEnd; }
  void setSegmentsBegin(Segment* segmentsBegin) {
    _segmentsBegin = segmentsBegin;
  }
  void setSegmentsEnd(Segment* segmentsEnd) { _segmentsEnd = segmentsEnd; }
  void setRootSegments();
  int getMaxBranchOrder();
  ~Neuron();

  private:
  int _layer;
  int _numberOfBranches;
  int _morphologicalType;
  int _electrophysiologicalType;
  int _neuronIndex;
  int _globalNeuronIndex;

  double _center[4];  // x,y,z, rotation y-axis

  Branch* _branch;
  Tissue* _neuronGroup;

  Segment* _segmentsBegin;
  Segment* _segmentsEnd;

  History* _translationHistory;
  History* _rotationYHistory;
};

#endif
