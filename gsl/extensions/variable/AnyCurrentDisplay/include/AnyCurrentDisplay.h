// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef AnyCurrentDisplay_H
#define AnyCurrentDisplay_H

#include "Mgs.h"
#include "CG_AnyCurrentDisplay.h"
#include <memory>
#include <fstream>

class AnyCurrentDisplay : public CG_AnyCurrentDisplay
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
                             CG_AnyCurrentDisplayInAttrPSet* CG_inAttrPset,
                             CG_AnyCurrentDisplayOutAttrPSet* CG_outAttrPset);
  AnyCurrentDisplay();
  virtual ~AnyCurrentDisplay();
  virtual void duplicate(std::unique_ptr<AnyCurrentDisplay>&& dup) const;
  virtual void duplicate(std::unique_ptr<Variable>&& dup) const;
  virtual void duplicate(std::unique_ptr<CG_AnyCurrentDisplay>&& dup) const;

  private:
  std::ofstream* outFile = 0;
  //std::string path;
};

#endif
