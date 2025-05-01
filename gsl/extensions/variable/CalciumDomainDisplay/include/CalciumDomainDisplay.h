// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef CalciumDomainDisplay_H
#define CalciumDomainDisplay_H

#include "CG_CalciumDomainDisplay.h"
#include "Mgs.h"
#include <fstream>
#include <memory>

class CalciumDomainDisplay : public CG_CalciumDomainDisplay
{
  public:
  void initialize(RNG& rng);
  void finalize(RNG& rng);
  virtual void dataCollection(Trigger* trigger, NDPairList* ndPairList);
  virtual void setUpPointers(
      const CustomString& CG_direction, const CustomString& CG_component,
      NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
      Constant* CG_constant, CG_CalciumDomainDisplayInAttrPSet* CG_inAttrPset,
      CG_CalciumDomainDisplayOutAttrPSet* CG_outAttrPset);
  CalciumDomainDisplay();
  virtual ~CalciumDomainDisplay();
  virtual void duplicate(std::unique_ptr<CalciumDomainDisplay>&& dup) const;
  virtual void duplicate(std::unique_ptr<Variable>&& dup) const;
  virtual void duplicate(std::unique_ptr<CG_CalciumDomainDisplay>&& dup) const;

  private:
  std::ofstream* outFile = 0;
};

#endif
