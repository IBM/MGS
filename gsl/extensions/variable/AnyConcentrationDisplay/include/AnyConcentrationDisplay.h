#ifndef AnyConcentrationDisplay_H
#define AnyConcentrationDisplay_H

#include "Lens.h"
#include "CG_AnyConcentrationDisplay.h"
#include <memory>
#include <fstream>

class AnyConcentrationDisplay : public CG_AnyConcentrationDisplay
{
  public:
  void initialize(RNG& rng);
  void finalize(RNG& rng);
  virtual void dataCollection(Trigger* trigger, NDPairList* ndPairList);
  virtual void setUpPointers(
      const String& CG_direction, const String& CG_component,
      NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
      Constant* CG_constant,
      CG_AnyConcentrationDisplayInAttrPSet* CG_inAttrPset,
      CG_AnyConcentrationDisplayOutAttrPSet* CG_outAttrPset);
  AnyConcentrationDisplay();
  virtual ~AnyConcentrationDisplay();
  virtual void duplicate(std::auto_ptr<AnyConcentrationDisplay>& dup) const;
  virtual void duplicate(std::auto_ptr<Variable>& dup) const;
  virtual void duplicate(std::auto_ptr<CG_AnyConcentrationDisplay>& dup) const;

  private:
  std::ofstream* outFile;
};

#endif
