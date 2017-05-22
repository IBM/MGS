#include "Lens.h"
#include "DetectDataChangeOneCompartment.h"
#include "CG_DetectDataChangeOneCompartment.h"
#include <memory>

void DetectDataChangeOneCompartment::initialize(RNG& rng) 
{
   if (not deltaT)
   {
      std::cerr << "ERROR: Please connect deltaT to " << typeid(*this).name() << std::endl;
   }
   assert(deltaT);
   if (not data)
   {
      std::cerr << "ERROR: Please connect Voltage|Calcium to " << typeid(*this).name() << std::endl;
   }
   assert(data);
   data_prev = (*data)[0];
}

void DetectDataChangeOneCompartment::calculateInfo(RNG& rng) 
{
   //slope = ((*data)[0] - data_prev)/(timeWindow);
   float prev_slope = slope;
   slope = ((*data)[0] - data_prev)/(data_prev);
   slope_absolute = fabs(slope);
   if (slope_absolute > criteria or 
         (prev_slope*slope < 0.0f))
   {// greater than threshold or 'pass a peak/nadir'
      data_prev = (*data)[0];
      timeWindow = 0.0;
      triggerWrite = 1;
   }
   else{
      triggerWrite = 0;
      timeWindow += (*deltaT);
   }
}

void DetectDataChangeOneCompartment::setUpPointers(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_DetectDataChangeOneCompartmentInAttrPSet* CG_inAttrPset, CG_DetectDataChangeOneCompartmentOutAttrPSet* CG_outAttrPset) 
{
   //VoltageArrayProducer* tmp = dynamic_cast<VoltageArrayProducer*>(data_connect);
   //if (tmp)
   //{

   //}
   //else{
   //   CaConcentrationArrayProducer* tmp = dynamic_cast<CaConcentrationArrayProducer*>(data_connect);
   //}
   data = data_connect;
}

DetectDataChangeOneCompartment::DetectDataChangeOneCompartment() 
   : CG_DetectDataChangeOneCompartment()
{
}

DetectDataChangeOneCompartment::~DetectDataChangeOneCompartment() 
{
}

void DetectDataChangeOneCompartment::duplicate(std::auto_ptr<DetectDataChangeOneCompartment>& dup) const
{
   dup.reset(new DetectDataChangeOneCompartment(*this));
}

void DetectDataChangeOneCompartment::duplicate(std::auto_ptr<Variable>& dup) const
{
   dup.reset(new DetectDataChangeOneCompartment(*this));
}

void DetectDataChangeOneCompartment::duplicate(std::auto_ptr<CG_DetectDataChangeOneCompartment>& dup) const
{
   dup.reset(new DetectDataChangeOneCompartment(*this));
}

