// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef AnyFluxDisplay_H
#define AnyFluxDisplay_H

#include "CG_AnyFluxDisplay.h"
#include "Mgs.h"
#include <fstream>
#include <memory>

class AnyFluxDisplay : public CG_AnyFluxDisplay
{
  public:
  void initialize(RNG& rng);
  void finalize(RNG& rng);
  virtual void dataCollection(Trigger* trigger, NDPairList* ndPairList);
  virtual void setUpPointers(const CustomString& CG_direction,
                             const CustomString& CG_component,
                             NodeDescriptor* CG_node, Edge* CG_edge,
                             VariableDescriptor* CG_variable,
                             Constant* CG_constant,
                             CG_AnyFluxDisplayInAttrPSet* CG_inAttrPset,
                             CG_AnyFluxDisplayOutAttrPSet* CG_outAttrPset);
  AnyFluxDisplay();
  virtual ~AnyFluxDisplay();
  virtual void duplicate(std::unique_ptr<AnyFluxDisplay>&& dup) const;
  virtual void duplicate(std::unique_ptr<Variable>&& dup) const;
  virtual void duplicate(std::unique_ptr<CG_AnyFluxDisplay>&& dup) const;

  private:
  std::ofstream* outFile = 0;
};

#endif
