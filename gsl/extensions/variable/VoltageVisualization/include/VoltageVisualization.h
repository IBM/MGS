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

#ifndef VoltageVisualization_H
#define VoltageVisualization_H

#include <mpi.h>
#include "Lens.h"
#include "CG_VoltageVisualization.h"
#include "SegmentDescriptor.h"
#include "CompartmentKey.h"
#include <map>
#include <list>
#include <memory>
#include <fstream>

class VoltageVisualization : public CG_VoltageVisualization
{
  public:
  VoltageVisualization();
  void initialize(RNG& rng);
  void finalize(RNG& rng);
  virtual void dataCollection(Trigger* trigger, NDPairList* ndPairList);
  virtual void setUpPointers(
      const String& CG_direction, const String& CG_component,
      NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
      Constant* CG_constant, CG_VoltageVisualizationInAttrPSet* CG_inAttrPset,
      CG_VoltageVisualizationOutAttrPSet* CG_outAttrPset);
  virtual ~VoltageVisualization();
  virtual void duplicate(std::auto_ptr<VoltageVisualization>& dup) const;
  virtual void duplicate(std::auto_ptr<Variable>& dup) const;
  virtual void duplicate(std::auto_ptr<CG_VoltageVisualization>& dup) const;

  private:
  float swapByteOrder(float*);

  FILE* _outFile;
  std::map<int, int> _fileOffsetMap;
  int _rank;
  int _size;
  std::vector<int> _ioNodes;
  bool _isIoNode;

  float** _voltageBuffs;
  int _nBuffs;
  int _nSends;
  int _nKreceives;
  int _nVreceives;

  // source size patterns of resident memory voltages
  std::vector<std::pair<dyn_var_t*, int> > _marshallPatterns;
  // rank buffer fwrite pattern consisting of data displacements and number
  std::vector<std::vector<std::vector<std::pair<int, int> > > >
      _demarshalPatterns;
  SegmentDescriptor _segmentDescriptor;
};

#endif
