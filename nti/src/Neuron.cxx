// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Neuron.h"

#include "Branch.h"
#include "Segment.h"
#include "Tissue.h"
#include "History.h"
#include "VecPrim.h"
#include <list>
#include <algorithm>
#include <float.h>

Neuron::Neuron()
    : _layer(0),
      _numberOfBranches(0),
      _morphologicalType(0),
      _electrophysiologicalType(0),
      _neuronIndex(0),
      _globalNeuronIndex(0),
      _branch(0),
      _neuronGroup(0),
      _segmentsBegin(0),
      _segmentsEnd(0),
      _translationHistory(0),
      _rotationYHistory(0)
{
}

Neuron::~Neuron()
{
  delete _translationHistory;
  delete _rotationYHistory;
  delete[] _branch;
}

void Neuron::setRootSegments()
{
  for (int j = 1; j < _numberOfBranches; ++j)
  {
    _branch[j].findRootSegment();
  }
}

void Neuron::resample(std::vector<Segment>& segments, double pointSpacing)
{
  for (int i = 0; i < _numberOfBranches; i++)
  {
    assert(_branch[i].getNumberOfSegments() > 0);
    _branch[i].resample(segments, pointSpacing);
  }
}

void Neuron::resetBranchRoots(std::vector<Segment>& segments)
{
  for (int i = 0; i < _numberOfBranches; i++)
  {
    assert(_branch[i].getNumberOfSegments() > 0);
    _branch[i].resetBranchRoots(segments);
  }
}

void Neuron::eliminateLostBranches()
{
  for (int i = 0; i < _numberOfBranches; i++)
  {
    int skip = 0;
    while (i + skip < _numberOfBranches &&
           _branch[i + skip].getNumberOfSegments() == 0)
      ++skip;
    if (skip > 0)
    {
      _numberOfBranches -= skip;
      for (int j = i; j < _numberOfBranches; ++j)
      {
        _branch[j] = _branch[j + skip];
        _branch[j].resetBranchIndex(j);
        int nsegs = _branch[j].getNumberOfSegments();
        Segment* segs = _branch[j].getSegments();
        for (int k = 0; k < nsegs; ++k)
        {
          segs[k].resetBranch(&_branch[j]);
        }
      }
    }
  }
}

Segment* Neuron::loadBinary(FILE* inputDataFile, Segment* segmentPtr,
                            Tissue* neuronGroup, const int neuronIndex,
                            FILE* translationHistoryFile,
                            FILE* rotationYHistoryFile)
{
  _segmentsBegin = segmentPtr;
  _neuronGroup = neuronGroup;
  _neuronIndex = neuronIndex;
  _globalNeuronIndex = _neuronGroup->getNeuronIndex() + _neuronIndex;

  if (translationHistoryFile)
    _translationHistory =
        new History(translationHistoryFile, _globalNeuronIndex, 3);
  if (rotationYHistoryFile)
    _rotationYHistory =
        new History(rotationYHistoryFile, _globalNeuronIndex, 1);

  // Read the center
  size_t s =
      fread(_center, sizeof(double[4]), 1, inputDataFile);  // sizeof(double[4])

  // Read the Layer
  s = fread(&_layer, sizeof(int), 1, inputDataFile);
  // Read the morphologicalType
  s = fread(&_morphologicalType, sizeof(int), 1, inputDataFile);
  // Read the electrophysiologicalType
  s = fread(&_electrophysiologicalType, sizeof(int), 1, inputDataFile);

  s = fread(&_numberOfBranches, sizeof(int), 1,
            inputDataFile);  // Finds the total Number of Branches
  _branch = new Branch[_numberOfBranches];  // Create the array of Branches

  for (int i = 0; i < _numberOfBranches; i++)  // Store the branches
    segmentPtr = _branch[i].loadBinary(inputDataFile, segmentPtr, this, i);

  _segmentsEnd = segmentPtr;
  return segmentPtr;
}

Segment* Neuron::loadText(FILE* inputDataFile, Segment* segmentPtr,
                          Tissue* neuronGroup, const int neuronIndex,
                          FILE* translationHistoryFile,
                          FILE* rotationYHistoryFile, int layer,
                          int morphologicalType, int electrophysiologicalType,
                          double xOffset, double yOffset, double zOffset,
                          char offsetType)
{
  _segmentsBegin = segmentPtr;
  _neuronGroup = neuronGroup;
  _neuronIndex = neuronIndex;
  _globalNeuronIndex = _neuronGroup->getNeuronIndex() + _neuronIndex;

  if (translationHistoryFile)
    _translationHistory =
        new History(translationHistoryFile, _globalNeuronIndex, 3);
  if (rotationYHistoryFile)
    _rotationYHistory =
        new History(rotationYHistoryFile, _globalNeuronIndex, 1);

  _layer = layer;
  _morphologicalType = morphologicalType;
  _electrophysiologicalType = electrophysiologicalType;
  _numberOfBranches = 0;

  int seg, branchType, parent, prevSeg = 0, prevBranchType = -1, cb = 0;
  float x, y, z, r;
  int pos = ftell(inputDataFile);
  std::list<int> branchTerminals;
  while (fscanf(inputDataFile, "%d %d %f %f %f %f %d", &seg, &branchType, &x,
                &y, &z, &r, &parent) != EOF)
  {
    if (parent == -1)
    {
      int tmpSeg, tmpParent, tmpBranchType, pos2 = ftell(inputDataFile);
      float tmpx = 0.0, tmpy = 0.0, tmpz = 0.0, tmpr = 0.0;
      do
      {
        ++cb;
        x += tmpx;
        y += tmpy;
        z += tmpz;
        r += tmpr;
        pos2 = ftell(inputDataFile);
        if (fscanf(inputDataFile, "%d %d %f %f %f %f %d", &tmpSeg,
                   &tmpBranchType, &tmpx, &tmpy, &tmpz, &tmpr,
                   &tmpParent) == EOF)
          break;
      } while (tmpBranchType == 1 && tmpParent == 1);
      fseek(inputDataFile, pos2, SEEK_SET);
      x /= double(cb);
      y /= double(cb);
      z /= double(cb);
      r /= double(cb);
      --cb;
      if (offsetType == 'A')
      {
        xOffset -= x;
        yOffset -= y;
        zOffset -= z;
      }
      else if (offsetType != 'R')
      {
        std::cerr << "Unrecognized offset type in tissue specification! "
                     "(Offsets ignored). Use \'R\' or \'A\'." << std::endl;
        xOffset = yOffset = zOffset = 0;
      }
      _center[0] = x + xOffset;
      _center[1] = y + yOffset;
      _center[2] = z + zOffset;
    }
    else
    {
      if (parent != 1) parent -= cb;
      seg -= cb;
      if (parent != prevSeg || prevBranchType != branchType)
      {
        branchTerminals.push_back(parent);
        branchTerminals.push_back(prevSeg);
      }
    }
    prevSeg = seg;
    prevBranchType = branchType;
  }
  branchTerminals.push_back(prevSeg);
  branchTerminals.sort();
  branchTerminals.unique();
  prevSeg = 0;
  prevBranchType = -1;
  fseek(inputDataFile, pos, SEEK_SET);
  while (fscanf(inputDataFile, "%d %d %f %f %f %f %d", &seg, &branchType, &x,
                &y, &z, &r, &parent) != EOF)
  {
    if (parent == -1)
    {
      int tmpSeg, tmpParent, tmpBranchType, pos2;
      float tmpx = 0.0, tmpy = 0.0, tmpz = 0.0, tmpr = 0.0;
      do
      {
        pos2 = ftell(inputDataFile);
        if (fscanf(inputDataFile, "%d %d %f %f %f %f %d", &tmpSeg,
                   &tmpBranchType, &tmpx, &tmpy, &tmpz, &tmpr,
                   &tmpParent) == EOF)
          break;
      } while (tmpBranchType == 1 && tmpParent == 1);
      fseek(inputDataFile, pos2, SEEK_SET);
    }
    else
    {
      if (parent > 1) parent -= cb;
      seg -= cb;
    }
    if (find(branchTerminals.begin(), branchTerminals.end(), seg) !=
        branchTerminals.end())
    {
      ++_numberOfBranches;
    }
    prevSeg = seg;
    prevBranchType = branchType;
  }
  _branch = new Branch[_numberOfBranches];  // Create the array of Branches
  prevSeg = 0;
  prevBranchType = -1;
  fseek(inputDataFile, pos, SEEK_SET);
  for (int i = 0; i < _numberOfBranches; i++)  // Store the branches
    segmentPtr =
        _branch[i].loadText(inputDataFile, segmentPtr, this, i, branchTerminals,
                            xOffset, yOffset, zOffset, cb);
  _segmentsEnd = segmentPtr;
  return segmentPtr;
}

void Neuron::writeCoordinates(FILE* fp)
{
  fwrite(&_center, sizeof(double[4]), 1, fp);
  fwrite(&_layer, sizeof(int), 1, fp);
  fwrite(&_morphologicalType, sizeof(int), 1, fp);
  fwrite(&_electrophysiologicalType, sizeof(int), 1, fp);
  fwrite(&_numberOfBranches, sizeof(int), 1, fp);
  for (int i = 0; i < _numberOfBranches; i++) _branch[i].writeCoordinates(fp);
}

void Neuron::translate(double xyz[3], int iteration)
{
  if (_translationHistory) _translationHistory->add(xyz, iteration);
  for (Segment* seg = _segmentsBegin; seg != _segmentsEnd; ++seg)
  {
    seg->getCoords()[0] += xyz[0];
    seg->getCoords()[1] += xyz[1];
    seg->getCoords()[2] += xyz[2];
  }
}

void Neuron::rotateY(double angle, int iteration)
{
  if (_rotationYHistory) _rotationYHistory->add(&angle, iteration);
  double angleSin = sin(DEG_RAD * angle);
  double angleCos = cos(DEG_RAD * angle);
  double nx, nz;

  for (Segment* seg = _segmentsBegin; seg != _segmentsEnd; ++seg)
  {
    seg->getCoords()[0] -= _center[0];
    // seg->getCoords()[1] -= _center[1];
    seg->getCoords()[2] -= _center[2];

    // rotation about the y-axis
    nx = seg->getCoords()[2] * angleSin + seg->getCoords()[0] * angleCos,
    nz = seg->getCoords()[2] * angleCos - seg->getCoords()[0] * angleSin;
    seg->getCoords()[0] = nx;
    seg->getCoords()[2] = nz;

    seg->getCoords()[0] += _center[0];
    // seg->getCoords()[1] += _center[1];
    seg->getCoords()[2] += _center[2];
  }
}

int Neuron::getMaxBranchOrder()
{
  int rval = -1;
  for (int i = 0; i < _numberOfBranches; ++i)
  {
    if (_branch[i].getBranchOrder() > rval) rval = _branch[i].getBranchOrder();
  }
  return rval;
}
