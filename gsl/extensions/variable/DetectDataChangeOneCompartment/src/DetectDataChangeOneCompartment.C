// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#include "Mgs.h"
#include "DetectDataChangeOneCompartment.h"
#include "CG_DetectDataChangeOneCompartment.h"
#include <memory>

#define SOFT_CONSTRAINT

void DetectDataChangeOneCompartment::initialize(RNG& rng) 
{
   if (not deltaT)
   {
      std::cerr << "ERROR: Please connect deltaT to " << typeid(*this).name() << std::endl;
   }
   assert(deltaT);
#if DETECT_CHANGE == _SINGLE_SENSOR_DETECT_CHANGE
   {
#ifdef SOFT_CONSTRAINT
   if (! data)
   {
      std::cerr << "WARNING: Please connect Voltage | Calcium to " << typeid(*this).name() << std::endl;
      return;
   }
#else
   //NOTE we set this to soft-constraint
   if (! data)
   {
      std::cerr << "ERROR: Please connect Voltage | Calcium to " << typeid(*this).name() << std::endl;
   }
#endif
   assert(data);
   data_prev = (*data)[0];
   }

#else
   {
#ifdef SOFT_CONSTRAINT
   if (data.size() == 0)
   {
      std::cerr << "WARNING: Please connect Voltage | Calcium to " << typeid(*this).name() << std::endl;
      return;
   }
#else
   if (data.size() == 0)
   {
      std::cerr << "ERROR: Please connect Voltage | Calcium to " << typeid(*this).name() << std::endl;
      assert(0);
   }
#endif
   data_prev.increaseSizeTo(data.size());
   slope.increaseSizeTo(data.size());
   for (int ii = 0; ii < data.size(); ii++)
      data_prev[ii] = (*(data[ii]))[0];
   }
#endif
}

bool DetectDataChangeOneCompartment::check_one_sensor(int ii) 
{
#ifdef SOFT_CONSTRAINT
   if (data.size() == 0)
      return false;
#endif
   bool sensor_triggered = false;
   float prev_slope = slope[ii];
   slope[ii] = ((*(data[ii]))[0] - data_prev[ii])/(data_prev[ii]);
   slope_absolute = fabs(slope[ii]);
   if (slope_absolute > criteria or 
         (prev_slope*slope[ii] < 0.0f))
   {// greater than threshold or 'pass a peak/nadir'
      sensor_triggered = true;
      //if (prev_slope*slope[ii] < 0.0f)
      //   pass_nadir_or_peak = true;
   }
   return sensor_triggered;
}
#if DETECT_CHANGE == _SINGLE_SENSOR_DETECT_CHANGE
void DetectDataChangeOneCompartment::calculateInfo(RNG& rng) 
{
#ifdef SOFT_CONSTRAINT
   if (! data)
      return ;
#endif
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
#else
void DetectDataChangeOneCompartment::calculateInfo(RNG& rng) 
{
#ifdef SOFT_CONSTRAINT
   if (data.size() == 0)
      return ;
#endif
   bool sensor_triggered = false;
   for (int ii = 0; ii < data.size(); ii++)
   {
      sensor_triggered = sensor_triggered or check_one_sensor(ii);
      if (sensor_triggered)
         break;
   }
   //if (sensor_triggered and (pass_nadir_or_peak or timeWindow >= temporal_resolution))
   //if (sensor_triggered and (timeWindow >= temporal_resolution))
   if (sensor_triggered)
   {// greater than threshold or 'pass a peak/nadir'
      for (int ii = 0; ii < data.size(); ii++)
         data_prev[ii] = (*(data[ii]))[0];
      timeWindow = 0.0;
      triggerWrite = 1;
      //pass_nadir_or_peak = false;
   }
   else{
      triggerWrite = 0;
      timeWindow += (*deltaT);
   }
}
#endif

void DetectDataChangeOneCompartment::setUpPointers(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_DetectDataChangeOneCompartmentInAttrPSet* CG_inAttrPset, CG_DetectDataChangeOneCompartmentOutAttrPSet* CG_outAttrPset) 
{
   //VoltageArrayProducer* tmp = dynamic_cast<VoltageArrayProducer*>(data_connect);
   //if (tmp)
   //{

   //}
   //else{
   //   CaConcentrationArrayProducer* tmp = dynamic_cast<CaConcentrationArrayProducer*>(data_connect);
   //}
#if DETECT_CHANGE == _SINGLE_SENSOR_DETECT_CHANGE
   data = data_connect;
#else
   data.push_back(data_connect);
#endif
}
DetectDataChangeOneCompartment::DetectDataChangeOneCompartment() 
   : CG_DetectDataChangeOneCompartment()
{
}

DetectDataChangeOneCompartment::~DetectDataChangeOneCompartment() 
{
}

void DetectDataChangeOneCompartment::duplicate(std::unique_ptr<DetectDataChangeOneCompartment>&& dup) const
{
   dup.reset(new DetectDataChangeOneCompartment(*this));
}

void DetectDataChangeOneCompartment::duplicate(std::unique_ptr<Variable>&& dup) const
{
   dup.reset(new DetectDataChangeOneCompartment(*this));
}

void DetectDataChangeOneCompartment::duplicate(std::unique_ptr<CG_DetectDataChangeOneCompartment>&& dup) const
{
   dup.reset(new DetectDataChangeOneCompartment(*this));
}

