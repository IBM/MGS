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

#ifndef VoltageDisplay_H
#define VoltageDisplay_H

#include "CG_VoltageDisplay.h"
#include "Lens.h"
#include <fstream>
#include <memory>

class VoltageDisplay : public CG_VoltageDisplay {
public:
  void initialize(RNG &rng);
  void finalize(RNG &rng);
  virtual void dataCollection(Trigger *trigger, NDPairList *ndPairList);
  virtual void setUpPointers(const String &CG_direction,
                             const String &CG_component,
                             NodeDescriptor *CG_node, Edge *CG_edge,
                             VariableDescriptor *CG_variable,
                             Constant *CG_constant,
                             CG_VoltageDisplayInAttrPSet *CG_inAttrPset,
                             CG_VoltageDisplayOutAttrPSet *CG_outAttrPset);
  VoltageDisplay();
  virtual ~VoltageDisplay();
  virtual void duplicate(std::auto_ptr<VoltageDisplay> &dup) const;
  virtual void duplicate(std::auto_ptr<Variable> &dup) const;
  virtual void duplicate(std::auto_ptr<CG_VoltageDisplay> &dup) const;

private:
  std::ofstream *outFile = 0;
};

#endif
