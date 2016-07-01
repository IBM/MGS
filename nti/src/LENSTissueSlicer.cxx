// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// ================================================================

#include "LENSTissueSlicer.h"
#include "Tissue.h"
#include "Neuron.h"
#include "Branch.h"
#include "Segment.h"
#include "Decomposition.h"
#include "AllInSegmentSpace.h"
#include "VecPrim.h"
#include "Params.h"
#include "Capsule.h"
#include "TissueContext.h"
#include "Sphere.h"
#include <math.h>
#include <cassert>
#include <list>
#include <algorithm>
#include <utility>
#include "MaxComputeOrder.h"

LENSTissueSlicer::LENSTissueSlicer(const int rank, const int nSlicers,
                                   const int nTouchDetectors,
                                   TissueContext* tissueContext, Params* params)
    : TissueSlicer(rank, nSlicers, nTouchDetectors, tissueContext->_tissue,
                   &tissueContext->_decomposition, params),
      _tissueContext(tissueContext),
      _sliced(false)
{
  _dataSize = N_CAP_DATA;
#ifdef A2AW
  Capsule capsule;
  int blocklen = _dataSize;

  MPI_Aint segAddress, disp;

  MPI_Get_address(&capsule, &segAddress);
  MPI_Get_address(capsule.getData(), &disp);

  disp -= segAddress;

  MPI_Datatype typeSegmentDataBasic;
  MPI_Type_create_hindexed(1, &blocklen, &disp, MPI_DOUBLE,
                           &typeSegmentDataBasic);
  MPI_Type_create_resized(typeSegmentDataBasic, 0, sizeof(Capsule),
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
  MPI_Type_contiguous(_dataSize, MPI_DOUBLE, _typeSegments);
  MPI_Type_commit(_typeSegments);
#endif
}

LENSTissueSlicer::~LENSTissueSlicer() {}

void LENSTissueSlicer::sliceAllNeurons()
{
  if (_tissueContext->_touchVector.getBlockCount() > 0 && !_sliced)
  {
    Decomposition* decomposition = *_decomposition;
    TouchVector::TouchIterator tend = _tissueContext->_touchVector.end();
    for (int i = 0; i < _numberOfReceivers; ++i)
      _sliceSegmentIndices[i].clear();
    //_sliceSegmentIndices[volumeIndex].push_back(i);
    for (TouchVector::TouchIterator titer =
             _tissueContext->_touchVector.begin();
         titer != tend; ++titer)
    {
      double key1 = titer->getKey1();
      double key2 = titer->getKey2();
      int c1Idx = _tissueContext->getCapsuleIndex(key1);
      int c2Idx = _tissueContext->getCapsuleIndex(key2);
      Capsule& c1 = _tissueContext->_capsules[c1Idx];
      Capsule& c2 = _tissueContext->_capsules[c2Idx];

      // Determine the volumeIndex of the key1 node's LENS touch element
      int v1Idx = -1;
      int rank2HandleCapsule;
      TissueContext::CapsuleAtBranchStatus status;
#ifdef IDEA1
      if (_tissueContext->isPartOfExplicitJunction(c1, *titer, status, rank2HandleCapsule))
      //if (_tissueContext->isPartOfExplicitJunction(c1, *titer, status, rank2HandleCapsule, decomposition))
      {
          v1Idx = rank2HandleCapsule;
      }
#else
      if (_segmentDescriptor.getFlag(key1) &&
          _tissueContext->isTouchToEnd(c1, *titer))
      {
        Sphere sphere;
        c1.getEndSphere(sphere);
        v1Idx = decomposition->getRank(sphere);
      }
#endif
      else
        v1Idx = decomposition->getRank(c1.getSphere());

      // Determine the volumeIndex of the key2 node's LENS touch element
      int v2Idx = -1;
#ifdef IDEA1
      if (_tissueContext->isPartOfExplicitJunction(c2, *titer, status, rank2HandleCapsule))
      //if (_tissueContext->isPartOfExplicitJunction(c2, *titer, status, rank2HandleCapsule, decomposition))
      {
          //v2Idx = _tissueContext->getJunctionMPIRank(c2);
          v2Idx = rank2HandleCapsule;
      }
#else
      if (_segmentDescriptor.getFlag(titer->getKey2()) &&
          _tissueContext->isTouchToEnd(c2, *titer))
      {
        Sphere sphere;
        c2.getEndSphere(sphere);
        v2Idx = decomposition->getRank(sphere);
      }
#endif
      else
        v2Idx = decomposition->getRank(c2.getSphere());

      if (v1Idx != _rank)
      {
        _sliceSegmentIndices[v1Idx].push_back(c1Idx);
        _sliceSegmentIndices[v1Idx].push_back(c2Idx);
        _tissueContext->_touchVector.mapTouch(v1Idx, titer);
      }
      /*
           //TUAN: fix this if 'neuron' decomposition is used
      else if (v2Idx!=_rank) // This else only required for neuron decomposition
        _sliceSegmentIndices[v1Idx].push_back(c2Idx);
      */
      if (v2Idx != v1Idx)
      {
        if (v2Idx != _rank)
        {
          _sliceSegmentIndices[v2Idx].push_back(c1Idx);
          _sliceSegmentIndices[v2Idx].push_back(c2Idx);
          _tissueContext->_touchVector.mapTouch(v2Idx, titer);
        }
        /*
        else if (v1Idx!=_rank) // This else only required for neuron
        decomposition
          _sliceSegmentIndices[v2Idx].push_back(c1Idx);
        */
      }
    }
    for (int i = 0; i < _numberOfReceivers; ++i)
    {
      sort(_sliceSegmentIndices[i].begin(), _sliceSegmentIndices[i].end());
      std::vector<long int>::iterator newEnd = unique(
          _sliceSegmentIndices[i].begin(), _sliceSegmentIndices[i].end());
      _sliceSegmentIndices[i].resize(newEnd - _sliceSegmentIndices[i].begin());
    }
  }
  _sliced = true;
}

void LENSTissueSlicer::writeBuff(int i, int j, int& writePos)
{
  assert(writePos <= _sendBuffSize - _dataSize);
  std::copy(_tissueContext->_capsules[(_sliceSegmentIndices[i])[j]].getData(),
            _tissueContext->_capsules[(_sliceSegmentIndices[i])[j]].getData() +
                _dataSize,
            &_sendBuff[writePos]);
  // memcpy(&_sendBuff[writePos],
  // _tissueContext->_capsules[(_sliceSegmentIndices[i])[j]].getData(),
  // _dataSize*sizeof(double));
  writePos += _dataSize;
}

void* LENSTissueSlicer::getSendBuff()
{
#ifdef A2AW
  return (void*)_tissueContext->_capsules;
#else
  return (void*)_sendBuff;
#endif
}

