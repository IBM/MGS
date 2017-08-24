// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "Lens.h"
#include "VoltageClamp.h"
#include "CG_VoltageClamp.h"
#include <memory>
#include <typeinfo>
#include <cmath>
#include "MaxComputeOrder.h"

#define SMALL 1.0e-6
// use 0 or 1
#define ABRUPT_JUMP_VOLTAGE 1

void VoltageClamp::initialize(RNG& rng) 
{
  if (not deltaT)
  {
    std::cerr << "ERROR: Please connect deltaT to " << typeid(*this).name() << std::endl;
  }
  if (not V)
  {
    std::cerr << "ERROR: Please connect Voltage to be clamped to " << typeid(*this).name() << std::endl;
  }
  assert(deltaT);
  assert(V);

  if (fileName.size() > 0)
  {
    outFile = new std::ofstream(fileName.c_str());
    (*outFile)<<"# type = " << type << "\n";
    //(*outFile)<<"# Time\tCurrent\n";
#ifdef USE_SERIES_RESISTANCE
    (*outFile)<<"# Time\tCurrent(pA)\tRs(GOhm)\ttargetV(mV)\tVoltage(mV)\n";
#else
    (*outFile)<<"# Time\tCurrent(pA)\tbeta\tCm(pF/um^2)\ttargetV(mV)\tVoltage(mV)\n";
#endif
  }

  //NOTE: If not defined, idx=0 default
#ifdef USE_SERIES_RESISTANCE
  //nothing to do
#else
  surfaceArea = &(dimensions[idx]->surface_area);
#endif
  _isOn = false;
  _status = VoltageClamp::NO_CLAMP;

  _Vprev = (*V)[idx];
  waveformIdx = waveform.size();
  if (gainTime < SMALL)
    gainTime = 0.005; // [ms]
    //gainTime = 0.05; // [ms]
  update_gainTime();
}

float VoltageClamp::getCurrentTime()
{
  return (getSimulation().getIteration()-1) * (*deltaT);
}

void VoltageClamp::update_gainTime()
{
#define DV_GAINTIME 100  // [mV] - the voltage change for the given 'gainTime'
  _gainTime = std::fabs(command - _Vstart) / (DV_GAINTIME) * gainTime;
}

void VoltageClamp::updateI(RNG& rng) 
{
  Igen = 0;
#ifdef CONSIDER_DI_DV
  conductance_didv = 0.0;
#endif
  bool inject=false;
  float targetV=0;
  if (_isOn) {
    targetV=command;
    inject=true;
  }
  if (waveformIdx<waveform.size()) {
    //assume each element map to the single time-step element
    targetV=waveform[waveformIdx];
    ++waveformIdx;
    inject=true;
    //HOWVER: This is not good as the result changes with the chosen time-step 
    //SO    : try type=3; where the dynamic voltage has associated time information so we can extrapolate the 
    //        data point and give the same result regardless of time-step being used
  }
  if (inject) {
    if (type == 1 or type == 3)
    {
      float goal;
      if (type == 1)
      {
        //float goal = (*V)[idx] + (targetV - (*V)[idx])/2;
#if ABRUPT_JUMP_VOLTAGE == 1
        //NOTE: assume abrupt jump 
        goal = targetV; 
#else
        float currentTime = getCurrentTime();
        if (_status == VoltageClamp::SLOPE_ON)
        {
          if (currentTime < _timeStart + _gainTime)
          {
            goal = _Vstart + (currentTime - _timeStart)/(_gainTime) * (targetV - _Vstart) ;
          }
          else{
            _status = VoltageClamp::FLAT_ZONE;
          }
        }
        else if (_status == VoltageClamp::SLOPE_OFF)
        {
          if (currentTime < _timeStart + _gainTime)
          {
            goal = targetV + (_timeStart + _gainTime - currentTime )/(_gainTime) * (_Vstart - targetV) ;
          }
          else{
            _status = VoltageClamp::FLAT_ZONE;
          }
        }
#endif
      }
      else if (type == 3)
      {
        //NOTE: goal = must be Vc interpolated at time (t+dt/2)
        // while (*V)[idx] is Vm at time (t) only
         assert(0);
         // update 'goal' here
      }
      double Igen_dv;
      
#ifdef USE_SERIES_RESISTANCE
      //NOTE: do we need to multiply surface area?
      //NOTE: V = I * R
      //     Volt = Ampere * Ohm
      //     mV   = Ampere * Ohm * 1e-3
      //     mV   = pA     * Ohm * 1e-3 * 1e+12
      //     mV   = pA     * GOhm * 1e-3 * 1e+12 * 1e-9
      //     mV   = pA     * GOhm   <---- correct
      // check unit
      //   Vm (mV)
      //   Igen (pA)
      //   Rs  (GOhm)
      Igen = ( ( goal - (*V)[idx] ) ) / Rs;
      Igen_dv =  ( goal - ((*V)[idx]+0.001 ) ) / Rs;
#else
      // Cm = pF/um^2 
      // I = dQ/dt = Q/t 
      // Q = (Coulombs) electric charge transfered through surface area over a time 
      // t = (second)
      // I = ampere
      // --> Ampere = Coulombs / second
      // NOTE: Coulomb = Farad * Volt
      //  -> Farad = Coulomb / Volt
      //     Farad*1e-12 = Coulomb*1e-12 / (Volt*1e+3) * 1e+3
      //     pF   = pico-Coulomb / (mV) * 1e+3
      //     --> pF/um^2 = pico-Coulomb/(um^2 * mV) * 1e+3
      //
      //  Igen (pA) = pico-Coulomb / (second)
      //            = pico-Coulomb / (ms) * 1e+3
      //            = [pico-Coulomb / (um^2 * mV) * 1e+3] * [um^2 * mV / ms]
      //            = Cm * (Voltage) / time_window * surface_area;
      //  beta = headstage gain (unitless)
      Igen = beta * Cm * ( ( goal - (*V)[idx] ) / (*deltaT/2) ) * *surfaceArea;
      Igen_dv = beta * Cm * ( ( goal - ((*V)[idx]+0.001) ) / (*deltaT/2) ) * *surfaceArea;
#endif
#ifdef CONSIDER_DI_DV
      double dI = Igen_dv - Igen; 
      conductance_didv = dI/(0.001);
#endif
      if (outFile)
      {
#ifdef USE_SERIES_RESISTANCE
        (*outFile)<<getCurrentTime()<<"\t"<<Igen
          <<"\t"<<Rs<<"\t"<<targetV<<"\t"<<(*V)[idx]<<"\n";
#else
        (*outFile)<<getCurrentTime()<<"\t"<<Igen
          <<"\t"<<beta<<"\t"<<Cm<<"\t"<<targetV<<"\t"<<(*V)[idx]<<" "<<"\n";
#endif
      }
    }
    else if (type == 2)
    {
      (*V)[idx] = targetV;
      float currentTime = getCurrentTime();
      if (_status == VoltageClamp::SLOPE_ON)
      {
        if (currentTime < _timeStart + _gainTime)
        {
          (*V)[idx] = _Vstart + (currentTime - _timeStart)/(_gainTime) * (targetV - _Vstart) ;
        }
        else{
          _status = VoltageClamp::FLAT_ZONE;
        }
      }
      else if (_status == VoltageClamp::SLOPE_OFF)
      {
        if (currentTime < _timeStart + _gainTime)
        {
          (*V)[idx] = targetV + (_timeStart + _gainTime - currentTime )/(_gainTime) * (_Vstart - targetV) ;
        }
        else{
          _status = VoltageClamp::FLAT_ZONE;
        }
      }
    }
    else{
      std::cerr << "Unsupported VoltgeClamp type" << std::endl;
    }
  }
  _Vprev = (*V)[idx];
}

void VoltageClamp::finalize(RNG& rng) 
{
  if (outFile)
    outFile->close();
}

void VoltageClamp::startWaveform(Trigger* trigger, NDPairList* ndPairList) 
{
  waveformIdx = 0;
}

void VoltageClamp::setCommand(Trigger* trigger, NDPairList* ndPairList) 
{
  NDPairList::iterator iter=ndPairList->begin();
  NDPairList::iterator end=ndPairList->end();
  for (; iter!=end; ++iter) {
    if ( (*iter)->getName() == "command" ) {
      command = static_cast<NumericDataItem*>((*iter)->getDataItem())->getFloat();
      _timeStart = getCurrentTime();
      _Vstart = (*V)[idx];
      if (_Vprev < command)
      {
        _status = VoltageClamp::SLOPE_ON;
      }
      else{
        _status = VoltageClamp::SLOPE_OFF;
      }
      update_gainTime();
    }
  }
}

void VoltageClamp::toggle(Trigger* trigger, NDPairList* ndPairList) 
{
  if (ndPairList == 0)
  {
    _isOn = not _isOn;
  }
  else{
    NDPairList::iterator iter=ndPairList->begin();
    NDPairList::iterator end=ndPairList->end();
    for (; iter!=end; ++iter) {
      if ( (*iter)->getName() == "toggle" ) {
        _isOn = (static_cast<NumericDataItem*>((*iter)->getDataItem())->getInt()>0) ? true : false;
      }
    }

  }
  if (_isOn)
  {
    _timeStart = getCurrentTime();
    _Vstart = (*V)[idx];
    if (_Vprev < command)
    {
      _status = VoltageClamp::SLOPE_ON;
    }
    else{
      _status = VoltageClamp::SLOPE_OFF;
    }
  }
  else{
    _status = VoltageClamp::NO_CLAMP;
  }
  waveformIdx=waveform.size();
}

VoltageClamp::VoltageClamp() 
  : CG_VoltageClamp(), outFile(0)
{
}

VoltageClamp::~VoltageClamp() 
{
  delete outFile;
}

void VoltageClamp::duplicate(std::auto_ptr<VoltageClamp>& dup) const
{
   dup.reset(new VoltageClamp(*this));
}

void VoltageClamp::duplicate(std::auto_ptr<Variable>& dup) const
{
   dup.reset(new VoltageClamp(*this));
}

void VoltageClamp::duplicate(std::auto_ptr<CG_VoltageClamp>& dup) const
{
   dup.reset(new VoltageClamp(*this));
}

