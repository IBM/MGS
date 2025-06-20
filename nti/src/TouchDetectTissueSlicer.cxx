// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "TouchDetectTissueSlicer.h"
#include "Tissue.h"
#include "Neuron.h"
#include "Branch.h"
#include "Segment.h"
#include "Decomposition.h"
#include "Capsule.h"
#include "VecPrim.h"
#include "Params.h"
#include "TissueContext.h"
#include "ComputeBranch.h"
#include "ShallowArray.h"
#include "TouchSpace.h"
#include <math.h>
#include <cassert>
#include <algorithm>

#define WRITE_SZ1 (N_SEG_DATA - 3)
#define WRITE_SZ2 3

TouchDetectTissueSlicer::TouchDetectTissueSlicer(
    const int rank, const int nSlicers, const int nTouchDetectors,
    Tissue* tissue, Decomposition** decomposition, TissueContext* tissueContext,
    Params* params, const int maxComputeOrder)
    : TissueSlicer(rank, nSlicers, nTouchDetectors, tissue, decomposition,
                   params),
      _maxComputeOrder(maxComputeOrder),
      _segs(0),
      _sendLostDaughters(false),
      _addCutPointJunctions(true),
      _tissueContext(tissueContext),
      _tolerance(0)
{
  assert(_params);
  if (_tissue->isEmpty())
    _segs = new Segment[1];
  else
    _segs = _tissue->getSegments();
  _dataSize = N_SEG_DATA;

#ifdef A2AW
  Segment segments[2];
  int blocklen[2] = {WRITE_SZ1, WRITE_SZ2};

  MPI_Aint segAddress;
  MPI_Aint disp[2];

  MPI_Get_address(&segments[0], &segAddress);
  MPI_Get_address(segments[0].getCoords(), &disp[0]);
  MPI_Get_address(segments[1].getCoords(), &disp[1]);

  disp[0] -= segAddress;
  disp[1] -= segAddress;

  MPI_Datatype typeSegmentDataBasic;
  MPI_Type_create_hindexed(2, blocklen, disp, MPI_DOUBLE,
                           &typeSegmentDataBasic);
  MPI_Type_create_resized(typeSegmentDataBasic, 0, sizeof(Segment),
                          &_typeSegmentData);
  MPI_Type_commit(&_typeSegmentData);

  for (int i = 0; i < _numberOfReceivers; ++i)
  {
    int numSegs = 1;
    MPI_Type_indexed(numSegs, _segmentBlockLengths, _segmentBlockDisplacements,
                     _typeSegmentData, &_typeSegments[i]);
    MPI_Type_commit(&_typeSegments[i]);
  }
#else
  MPI_Type_contiguous(_dataSize, MPI_DOUBLE, &_typeSegments[0]);
  MPI_Type_commit(&_typeSegments[0]);
#endif
}

TouchDetectTissueSlicer::~TouchDetectTissueSlicer()
{
  if (_tissue->isEmpty()) delete[] _segs;
}

//GOAL: work with Segment, Branch
//   find out which Segment point is parent-end of the Junction
//   (i.e. set the flag)
void TouchDetectTissueSlicer::sliceAllNeurons()
{
  if (!_tissue->isEmpty())
  {
    for (int i = 0; i < _numberOfReceivers; ++i)
      _sliceSegmentIndices[i].clear();
    Segment* previousSeg = _segs;
    Branch* branch, * previousBranch = 0;
    int idx = -1;
    int computeOrder = 0;
    int segVolumeIndex, prevSegVolumeIndex = -1;
    int rootVolumeIndex;
    double radius = 0;
    Decomposition* decomposition = *_decomposition;

    ShallowArray<int, MAXRETURNRANKS, 100> indices;
    Segment* segsEnd = _segs + _tissue->getSegmentArraySize() - 1;
    for (Segment* seg = _segs; seg <= segsEnd; ++seg)
    {
      branch = seg->getBranch();
      segVolumeIndex = decomposition->getRank(seg->getSphere());//the Segment obj belong
        //the MPI rank that hold the segment's sphere

      if (branch == previousBranch)
      {//new branch or continued the branch
        assert(_params);
        decomposition->addRanks(
            &previousSeg->getSphere(), seg->getCoords(),
            _params->getRadius(previousSeg->getSegmentKey()) + _tolerance,
            indices);// return all the MPI ranks in 'indices' at which the given Capsule
        // as defined by the starting sphere (of previous Segment), and the 
        // current sphere (of current Segment) with a given tolerance '_tolerance'
        // is intersectd with, i.e. such ranks are supposed to 
        // have the Segment (which will be used for TouchDetection)
        if (_sendLostDaughters && previousSeg->getSegmentIndex() == 0 &&
            branch->getBranchOrder() != 0)
        {
          // send lost daughters
          Segment* parentPreviousSeg = branch->getRootSegment() - 1;
          indices.push_back(
              decomposition->getRank(parentPreviousSeg->getSphere()));
        }
        indices.sort();
        indices.unique();
        ShallowArray<int, MAXRETURNRANKS, 100>::iterator iter,
            end = indices.end();
        for (iter = indices.begin(); iter != end; ++iter)
          _sliceSegmentIndices[(*iter)].push_back(idx);
      }
      indices.clear();

      // TUAN NOTE: A segment is a point, i.e. the core of a sphere
      // TUAN TODO: Remove the flag out of the key once IDEA1 is tested  working 
      seg->isJunctionSegment(false);
      if (branch != previousBranch)
      {//start a new branch
        if (branch->getBranchOrder() == 0)
        {
          seg->isJunctionSegment(true);
          computeOrder = _maxComputeOrder;
        }
        else
        {
          Segment* parentPreviousSeg = branch->getRootSegment() - 1;
          computeOrder = branch->getRootSegment()->getComputeOrder() + 1;
          if (computeOrder > _maxComputeOrder)
            parentPreviousSeg->isJunctionSegment(true);
        }
      }
      else if (segVolumeIndex != prevSegVolumeIndex &&
               seg->getSegmentIndex() <
                   seg->getBranch()->getNumberOfSegments() - 1 &&
               _addCutPointJunctions)
      {//same branch, but start a slicing cut
        ++computeOrder;
        if (computeOrder > _maxComputeOrder)
          previousSeg->isJunctionSegment(true);
        if (_sendLostDaughters)  // lost cut daughter idx++ now sent here
          indices.push_back(decomposition->getRank(previousSeg->getSphere()));
      }
      if (computeOrder > _maxComputeOrder) computeOrder = 0;
      seg->setComputeOrder(computeOrder);

      previousSeg = seg;
      previousBranch = branch;
      prevSegVolumeIndex = segVolumeIndex;
      ++idx;  // index of previousSeg
    }
    for (Segment* seg = _segs; seg <= segsEnd; ++seg) seg->setKey();
  }
}

void TouchDetectTissueSlicer::writeBuff(int i, int j, int& writePos)
{
  assert(writePos <= _sendBuffSize - (WRITE_SZ1 + WRITE_SZ2));
  std::copy(_segs[(_sliceSegmentIndices[i])[j]].getCoords(),
            _segs[(_sliceSegmentIndices[i])[j]].getCoords() + WRITE_SZ1,
            &_sendBuff[writePos]);
  // memcpy(&_sendBuff[writePos],
  // _segs[(_sliceSegmentIndices[i])[j]].getCoords(), WRITE_SZ1*sizeof(double));
  writePos += WRITE_SZ1;
  std::copy(_segs[(_sliceSegmentIndices[i])[j] + 1].getCoords(),
            _segs[(_sliceSegmentIndices[i])[j] + 1].getCoords() + WRITE_SZ2,
            &_sendBuff[writePos]);
  // memcpy(&_sendBuff[writePos],
  // _segs[(_sliceSegmentIndices[i])[j]+1].getCoords(),
  // WRITE_SZ2*sizeof(double));
  writePos += WRITE_SZ2;
}

void* TouchDetectTissueSlicer::getSendBuff()
{
#ifdef A2AW
  return (void*)_segs;
#else
  return (void*)_sendBuff;
#endif
}
