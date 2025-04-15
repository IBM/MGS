// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef CalciumDisplay_H
#define CalciumDisplay_H

#include "Lens.h"
#include "CG_CalciumDisplay.h"
#include <memory>
#include <fstream>

class CalciumDisplay : public CG_CalciumDisplay
{
  public:
  void initialize(RNG& rng);
  void finalize(RNG& rng);
  virtual void dataCollection(Trigger* trigger, NDPairList* ndPairList);
  virtual void setUpPointers(const String& CG_direction,
                             const String& CG_component,
                             NodeDescriptor* CG_node, Edge* CG_edge,
                             VariableDescriptor* CG_variable,
                             Constant* CG_constant,
                             CG_CalciumDisplayInAttrPSet* CG_inAttrPset,
                             CG_CalciumDisplayOutAttrPSet* CG_outAttrPset);
  CalciumDisplay();
  virtual ~CalciumDisplay();
  virtual void duplicate(std::unique_ptr<CalciumDisplay>& dup) const;
  virtual void duplicate(std::unique_ptr<Variable>& dup) const;
  virtual void duplicate(std::unique_ptr<CG_CalciumDisplay>& dup) const;

  private:
  std::ofstream* outFile = 0;
};

#endif
