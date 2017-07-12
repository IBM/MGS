// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef VOLUMEDIVIDER_H
#define VOLUMEDIVIDER_H

#include "Copyright.h"

#include <list>
#include <vector>
#include <deque>
#include <cassert>

class VolumeDivider
{

   public:
      VolumeDivider();
      
      virtual void setUp(std::vector<int>& dimensions, unsigned numGranules);
      bool simpleMethod() const {return _simpleMethod;}
      unsigned getPiece(const std::vector<int>& coordinates) const;      
      unsigned getSubPiece(int dimension, int coordinate) const;
      void getPieceCoordinates(unsigned partition, std::vector<double>& coordinates);
      std::vector<int> const & getDimensions() const {return _dimensions;}
      std::vector<int> const & getStrides() const {return _strides;}
      std::vector<int> const & getDividers() const {return _dividers;}
      std::vector<int> const & getMinChunkSizes() const {return _minChunkSizes;}
      std::vector<int> const & getCutOffs() const {return _cutOffs;}
      std::vector<int> const & getDimOrder() const {return _dimOrder;}
      std::vector<int> const & getRemainders() const {return _remainders;}
      unsigned getNumberOfPieces() {return _numPieces;}
      virtual ~VolumeDivider();

   private:
      bool _simpleMethod;

      void findPieces();
      void decompose(int number, std::deque<int>& factors) const;
      int indexMaxAvgChunk() const;

      std::vector<int> _dimensions;
      std::vector<int> _minChunkSizes;
      std::vector<int> _dividers;
      std::vector<int> _remainders; // is also the cutOff partition
      std::vector<int> _cutOffs;
      std::vector<int> _strides;
      std::vector<int> _dimOrder;
      int _numPieces;
};
#endif
