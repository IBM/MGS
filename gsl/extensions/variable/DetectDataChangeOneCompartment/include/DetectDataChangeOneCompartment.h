#ifndef DetectDataChangeOneCompartment_H
#define DetectDataChangeOneCompartment_H

#include "Lens.h"
#include "CG_DetectDataChangeOneCompartment.h"
#include <memory>
#include "MaxComputeOrder.h"

class DetectDataChangeOneCompartment : public CG_DetectDataChangeOneCompartment
{
   public:
      void initialize(RNG& rng);
      void calculateInfo(RNG& rng);
      bool check_one_sensor(int ii);

      virtual void setUpPointers(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_DetectDataChangeOneCompartmentInAttrPSet* CG_inAttrPset, CG_DetectDataChangeOneCompartmentOutAttrPSet* CG_outAttrPset);
      DetectDataChangeOneCompartment();
      virtual ~DetectDataChangeOneCompartment();
      virtual void duplicate(std::unique_ptr<DetectDataChangeOneCompartment>& dup) const;
      virtual void duplicate(std::unique_ptr<Variable>& dup) const;
      virtual void duplicate(std::unique_ptr<CG_DetectDataChangeOneCompartment>& dup) const;
   private:
      //bool pass_nadir_or_peak(false);
};

#endif
