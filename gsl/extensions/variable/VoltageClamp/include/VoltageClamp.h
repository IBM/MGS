// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef VoltageClamp_H
#define VoltageClamp_H

#include "Mgs.h"
#include "CG_VoltageClamp.h"
#include <fstream>
#include <memory>

class VoltageClamp : public CG_VoltageClamp
{
   public:
      enum VCLAMP_SLOPE { SLOPE_ON=0, FLAT_ZONE = 1, SLOPE_OFF=2, NO_CLAMP=3 };
      void initialize(RNG& rng);
      void updateI(RNG& rng);
      void finalize(RNG& rng);
      virtual void startWaveform(Trigger* trigger, NDPairList* ndPairList);
      virtual void setCommand(Trigger* trigger, NDPairList* ndPairList);
      virtual void toggle(Trigger* trigger, NDPairList* ndPairList);
      VoltageClamp();
      virtual ~VoltageClamp();
      virtual void duplicate(std::unique_ptr<VoltageClamp>&& dup) const;
      virtual void duplicate(std::unique_ptr<Variable>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_VoltageClamp>&& dup) const;
      virtual void setInjectedCurrent(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_VoltageClampInAttrPSet* CG_inAttrPset, CG_VoltageClampOutAttrPSet* CG_outAttrPset);
   private:
      std::ofstream* outFile = 0;
      float _timeStart; //time point when start Vclamp
      float _Vstart;  //voltage at timeStart
      int _status; // 
      bool _isOn;
      float _Vprev;	// Declare voltage from previous iteration for calculating dV
      int waveformIdx;  // Declare index of the waveform array
      float _gainTime; //[ms] the time to reach Vc2 from Vc1
      float getCurrentTime();
      void update_gainTime();
      std::map<std::string, std::vector<float> > data_timeVm;
      unsigned int time_index;
      void updateI_type3(RNG& rng);
      unsigned int num_rows;
      float _time_for_io;
      void do_IO(float targetV);
};

#endif
