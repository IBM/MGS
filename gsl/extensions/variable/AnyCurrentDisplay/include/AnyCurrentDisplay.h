#ifndef AnyCurrentDisplay_H
#define AnyCurrentDisplay_H

#include "Lens.h"
#include "CG_AnyCurrentDisplay.h"
#include <memory>
#include <fstream>

class AnyCurrentDisplay : public CG_AnyCurrentDisplay
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
                             CG_AnyCurrentDisplayInAttrPSet* CG_inAttrPset,
                             CG_AnyCurrentDisplayOutAttrPSet* CG_outAttrPset);
  AnyCurrentDisplay();
  virtual ~AnyCurrentDisplay();
  virtual void duplicate(std::auto_ptr<AnyCurrentDisplay>& dup) const;
  virtual void duplicate(std::auto_ptr<Variable>& dup) const;
  virtual void duplicate(std::auto_ptr<CG_AnyCurrentDisplay>& dup) const;

  private:
  std::ofstream* outFile = 0;
};

#endif
