// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

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
      const String& CG_direction, const String& CG_component,
      NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
      Constant* CG_constant, CG_CalciumVisualizationInAttrPSet* CG_inAttrPset,
      CG_CalciumVisualizationOutAttrPSet* CG_outAttrPset);
  virtual ~CalciumVisualization();
  virtual void duplicate(std::unique_ptr<CalciumVisualization>& dup) const;
  virtual void duplicate(std::unique_ptr<Variable>& dup) const;
  virtual void duplicate(std::unique_ptr<CG_CalciumVisualization>& dup) const;

  private:
  float swapByteOrder(float*);

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
