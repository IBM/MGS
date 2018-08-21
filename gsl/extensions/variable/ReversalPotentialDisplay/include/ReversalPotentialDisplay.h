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
      const String& CG_direction, const String& CG_component,
      NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
      Constant* CG_constant,
      CG_ReversalPotentialDisplayInAttrPSet* CG_inAttrPset,
      CG_ReversalPotentialDisplayOutAttrPSet* CG_outAttrPset);
  ReversalPotentialDisplay();
  virtual ~ReversalPotentialDisplay();
  virtual void duplicate(std::auto_ptr<ReversalPotentialDisplay>& dup) const;
  virtual void duplicate(std::auto_ptr<Variable>& dup) const;
  virtual void duplicate(std::auto_ptr<CG_ReversalPotentialDisplay>& dup) const;

  private:
  std::ofstream* outFile = 0;
};

#endif
