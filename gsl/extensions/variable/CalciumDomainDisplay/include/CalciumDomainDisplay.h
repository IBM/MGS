#ifndef CalciumDomainDisplay_H
#define CalciumDomainDisplay_H

#include "CG_CalciumDomainDisplay.h"
#include "Lens.h"
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
