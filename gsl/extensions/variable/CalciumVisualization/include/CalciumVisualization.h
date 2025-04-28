// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef CalciumVisualization_H
#define CalciumVisualization_H

#include <mpi.h>
#include "Lens.h"
#include "CG_CalciumVisualization.h"
#include "SegmentDescriptor.h"
#include "CompartmentKey.h"
#include <map>
#include <list>
#include <memory>
#include <fstream>

class CalciumVisualization : public CG_CalciumVisualization
{
  public:
  CalciumVisualization();
  void initialize(RNG& rng);
  void finalize(RNG& rng);
  virtual void dataCollection(Trigger* trigger, NDPairList* ndPairList);
  virtual void setUpPointers(
      const CustomString& CG_direction, const CustomString& CG_component,
      NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
      Constant* CG_constant, CG_CalciumVisualizationInAttrPSet* CG_inAttrPset,
      CG_CalciumVisualizationOutAttrPSet* CG_outAttrPset);
  virtual ~CalciumVisualization();
  virtual void duplicate(std::unique_ptr<CalciumVisualization>&& dup) const;
  virtual void duplicate(std::unique_ptr<Variable>&& dup) const;
  virtual void duplicate(std::unique_ptr<CG_CalciumVisualization>&& dup) const;

  private:
  void swapByteOrder(float*);

  FILE* _outFile;
  std::map<int, int> _fileOffsetMap;
  int _rank;
  int _size;
  std::vector<int> _ioNodes;
  bool _isIoNode;

  float** _calciumBuffs;
  int _nBuffs;
  int _nSends;
  int _nKreceives;
  int _nVreceives;

  // source size patterns of resident memory calcium concentrations
  std::vector<std::pair<dyn_var_t*, int> > _marshallPatterns;
  // rank buffer fwrite pattern consisting of data displacements and number
  std::vector<std::vector<std::vector<std::pair<int, int> > > >
      _demarshalPatterns;
  SegmentDescriptor _segmentDescriptor;
};

#endif
