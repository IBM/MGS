// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef ReversalPotentialDisplay_H
#define ReversalPotentialDisplay_H

#include "Lens.h"
#include "CG_ReversalPotentialDisplay.h"
#include <memory>
#include <fstream>

class ReversalPotentialDisplay : public CG_ReversalPotentialDisplay
{
  public:
  void initialize(RNG& rng);
  void finalize(RNG& rng);
  virtual void dataCollection(Trigger* trigger, NDPairList* ndPairList);
  virtual void setUpPointers(
      const CustomString& CG_direction, const CustomString& CG_component,
      NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
      Constant* CG_constant,
      CG_ReversalPotentialDisplayInAttrPSet* CG_inAttrPset,
      CG_ReversalPotentialDisplayOutAttrPSet* CG_outAttrPset);
  ReversalPotentialDisplay();
  virtual ~ReversalPotentialDisplay();
  virtual void duplicate(std::unique_ptr<ReversalPotentialDisplay>&& dup) const;
  virtual void duplicate(std::unique_ptr<Variable>&& dup) const;
  virtual void duplicate(std::unique_ptr<CG_ReversalPotentialDisplay>&& dup) const;

  private:
  std::ofstream* outFile = 0;
};

#endif
