// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef CurrentDisplay_H
#define CurrentDisplay_H

#include "CG_CurrentDisplay.h"
#include "Mgs.h"
#include <fstream>
#include <memory>

class CurrentDisplay : public CG_CurrentDisplay {
public:
  void initialize(RNG &rng);
  void finalize(RNG &rng);
  virtual void dataCollection(Trigger *trigger, NDPairList *ndPairList);
  virtual void setUpPointers(const CustomString &CG_direction,
                             const CustomString &CG_component,
                             NodeDescriptor *CG_node, Edge *CG_edge,
                             VariableDescriptor *CG_variable,
                             Constant *CG_constant,
                             CG_CurrentDisplayInAttrPSet *CG_inAttrPset,
                             CG_CurrentDisplayOutAttrPSet *CG_outAttrPset);
  CurrentDisplay();
  virtual ~CurrentDisplay();
  virtual void duplicate(std::unique_ptr<CurrentDisplay>&& dup) const;
  virtual void duplicate(std::unique_ptr<Variable>&& dup) const;
  virtual void duplicate(std::unique_ptr<CG_CurrentDisplay>&& dup) const;

private:
  std::ofstream *outFile = 0;
};

#endif
