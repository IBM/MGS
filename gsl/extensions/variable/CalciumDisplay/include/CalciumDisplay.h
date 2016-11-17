// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

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
  virtual void duplicate(std::auto_ptr<CalciumDisplay>& dup) const;
  virtual void duplicate(std::auto_ptr<Variable>& dup) const;
  virtual void duplicate(std::auto_ptr<CG_CalciumDisplay>& dup) const;

  private:
  std::ofstream* outFile;
};

#endif
