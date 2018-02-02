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
#include <algorithm>
#include <iomanip>
#include "MaxComputeOrder.h"
#include "FileUtils.h"
#include "NumberUtils.h"

#define SMALL 1.0e-6
// use 0 or 1
#define ABRUPT_JUMP_VOLTAGE 1
#define IO_INTERVAL 1.0 // ms
#define dvx 0.001   // (mV)
#define decimal_places 3
#define fieldDelimiter "\t"
#define Rs_tight 2.0e-4   //GOhm

// NOTE:
// type = 1
//
// type = 2
//
// type = 3
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
    outFile->precision(decimal_places);
    (*outFile)<<"# type = " << type << "\n";
    //(*outFile)<<"# Time\tCurrent\n";
    if (type == 1)
    {
#ifdef USE_SERIES_RESISTANCE
      //(*outFile)<<"# Time(ms)\tIgen(pA)\tIchan(pA)\tRs(MOhm)\ttargetV(mV)\tVoltage(mV)\n";
      (*outFile)<<"# Time(ms)\tIgen(pA)\tRs(MOhm)\ttargetV(mV)\tVoltage(mV)\n";
#else
      (*outFile)<<"# Time\tCurrent(pA)\tbeta\tCm(pF/um^2)\ttargetV(mV)\tVoltage(mV)\n";
#endif
    }
    else if (type == 2)
    {
      (*outFile)<<"# Time\ttargetV(mV)\tVoltage(mV)\n";
    }
  }


  if (type == 3)
  {
    if (tVm_file.size() == 0)
    {
      std::cerr << "ERROR: Please pass t-Vm file to 'tVm_file' argument when using type=3 VoltageClamp" << std::endl;
      assert(0);
    }
    else{
      std::string file_name(tVm_file.c_str());
      if (not FileFolderUtils::isFileExist(file_name))
      {
        std::cerr << "ERROR: Please pass file " << tVm_file << " exist when using type=3 VoltageClamp" << std::endl;
        assert(0);
      }
      unsigned int num_col;
      data_timeVm = FileFolderUtils::readCsvFile(file_name, num_col);
      time_index = 0;
      num_rows = data_timeVm["time"].size();
      std::vector<float>::iterator  iiter= data_timeVm["time"].begin(),
        iend = data_timeVm["time"].end();
      float offset = data_timeVm["time"][0];
      for (int i = 0; iiter < iend; iiter++, i++)
      {
        data_timeVm["time"][i] -= offset;
      }
    }
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
  _Vstart = (*V)[idx];
  waveformIdx = waveform.size();
  if (output_interval < SMALL)
    output_interval = IO_INTERVAL;  // (ms)
  if (gainTime < SMALL)
    gainTime = 0.005; // [ms]
    //gainTime = 0.05; // [ms]
  update_gainTime();
  _time_for_io = getCurrentTime();
  //Ichan = 0.0; // (pA)
}

float VoltageClamp::getCurrentTime()
{
  return ((double)getSimulation().getIteration()-1) * (*deltaT);
}

//GOAL: calculate the total amount of time to reach the new clamping value
//   given the assumption that for DV_GAINTIME=+100mV change, it takes 'gainTime' (ms)
void VoltageClamp::update_gainTime()
{
#define DV_GAINTIME 100  // [mV] - the voltage change for the given 'gainTime'
  _gainTime = std::fabs(command - _Vstart) / (DV_GAINTIME) * gainTime;
  std::cout << "gainTime is " << _gainTime << "(ms)";
}

void VoltageClamp::updateI_type3(RNG& rng)
{
  volatile float targetV=0.0;
  std::vector<float>::iterator low =
    std::lower_bound(data_timeVm["time"].begin(), data_timeVm["time"].end(), getCurrentTime());
  volatile int index = low - data_timeVm["time"].begin();
  assert(index>=0);
  if (index == 0)
    targetV = data_timeVm["Vm"][index];
  else if (index < num_rows)
    targetV = linear_interp(data_timeVm["time"][index-1], data_timeVm["Vm"][index-1],
        data_timeVm["time"][index], data_timeVm["Vm"][index], getCurrentTime());
  else //assume saturation in taum when Vm > max-value
     targetV = data_timeVm["Vm"][index-1];
  _Vprev = (*V)[idx];
  (*V)[idx] = targetV;

  do_IO(targetV);
  float goal = targetV;
#ifdef USE_SERIES_RESISTANCE
  {
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
      //Ichan = Igen
      //  - (surfaceArea * Csc * (goal - (*V)[idx]) / (*deltaT/2)) /*capacitive current*/
      //  - (surfaceArea * gLeak * ((*V)[idx] -Eleak)) /*leak current*/ ;
  }
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
      Igen_dv = beta * Cm * ( ( goal - ((*V)[idx]+dvx) ) / (*deltaT/2) ) * *surfaceArea;
#endif
#ifdef CONSIDER_DI_DV
      //double dI = Igen_dv - Igen;
      //Igen_dv =  ( goal - ((*V)[idx]+dvx) ) / Rs;
      //conductance_didv = std::abs(dI/(dvx));
      conductance_didv = 1.0/ Rs;  // ((nS))
#endif
  return;
}

void VoltageClamp::updateI(RNG& rng)
{
  if (type == 3)
  {
    updateI_type3(rng);
    return;
  }
  /////////////////////////////
#ifdef CONSIDER_DI_DV
  conductance_didv = 0.0;
#endif
  bool inject=false;
  float targetV=0;
  if (_isOn) {
    targetV=command;
    inject=true;
  }
  Igen = 0;
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
          if ((*V)[idx] > targetV)
          {
            if (currentTime < _timeStart + _gainTime)
            {
              //goal = targetV + (_timeStart + _gainTime - currentTime )/(_gainTime) * (_Vstart - targetV) ;
#define MAX_ALLOWED_DV 0.5
              goal = (*V)[idx] - std::min(MAX_ALLOWED_DV, (*V)[idx]-targetV);
            }
            else{
              _status = VoltageClamp::FLAT_ZONE;
            }
          }
        }
#endif
      }
      //else if (type == 3)
      //{
      //  //NOTE: goal = must be Vc interpolated at time (t+dt/2)
      //  // while (*V)[idx] is Vm at time (t) only
      //   assert(0);
      //   // update 'goal' here
      //}

#ifdef USE_SERIES_RESISTANCE
      //NOTE: do we need to multiply surface area?
      //NOTE: V = I * R
      //     Volt = Ampere * Ohm
      //     mV   = Ampere * Ohm * 1e-3
      //     mV   = pA     * Ohm * 1e-3 * 1e+12
      //            10^12(pA) * 10^{-9} GOhm * 1e-3
      //     mV   = pA     * GOhm * 1e-3 * 1e+12 * 1e-9
      //     mV   = pA     * GOhm   <---- correct
      // check unit
      //   Vm (mV)
      //   Igen (pA)
      //   Rs  (GOhm)
      Igen = ( ( goal - (*V)[idx] ) ) / Rs;  // (pA)
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
      Igen_dv = beta * Cm * ( ( goal - ((*V)[idx]+dvx) ) / (*deltaT/2) ) * *surfaceArea;
#endif
#ifdef CONSIDER_DI_DV
      conductance_didv = 1.0 / Rs; // (nS)
#endif
      do_IO(targetV);
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
      do_IO(targetV);
    }
    else{
      std::cerr << "Unsupported VoltgeClamp type" << std::endl;
    }
  }
  else{
      //NOTE: No Voltage-clamp I/O
      /*
    if (type == 1 or type == 3)
    {
      if (outFile)
      {
#ifdef USE_SERIES_RESISTANCE
        (*outFile)<<getCurrentTime()<<"\t"<<Igen
          <<"\t"<<Rs*1e3<<"\t"<<targetV<<"\t"<<(*V)[idx]<<"\n";
#else
        (*outFile)<<getCurrentTime()<<"\t"<<Igen
          <<"\t"<<beta<<"\t"<<Cm<<"\t"<<targetV<<"\t"<<(*V)[idx]<<" "<<"\n";
#endif
      }
    }
    else if (type == 2)
    {
      if (outFile)
      {
        (*outFile)<<getCurrentTime()<<"\t"<<
          targetV<<"\t"<<(*V)[idx]<<"\n";
      }
    }
       */
  }
  _Vprev = (*V)[idx];
}

void VoltageClamp::finalize(RNG& rng)
{
  if (outFile)
    outFile->close();
}

// GOAL: user pass in an array of Vm values, each value map to value for
// one iteration
void VoltageClamp::startWaveform(Trigger* trigger, NDPairList* ndPairList)
{
  waveformIdx = 0;
}

// GOAL: passing a new VClamp value via the 'command' argument
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

//GOAL : toggle the status of clamping ON/OF
//   default: toggle the current status
//   user can specify exactly what it should be via toggle=1 or toggle=0
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

void VoltageClamp::do_IO(float targetV)
{
  if (type == 1)
  {
    if (outFile)
    {
      if (getCurrentTime() > _time_for_io)
      {
        _time_for_io = getCurrentTime() + output_interval;
#ifdef USE_SERIES_RESISTANCE
        (*outFile) << std::fixed
          << std::setw(8) << getCurrentTime()<<"\t"
          << std::setw(8) << Igen <<"\t"
          //<< std::setw(8) << Ichan <<"\t"
          << std::setw(8) <<Rs*1e3<<"\t"
          << std::setw(8) <<targetV<<"\t"
          << std::setw(8) <<(*V)[idx]<<"\n";
#else
        (*outFile)<< std::fixed << getCurrentTime()<<"\t"<<Igen
          <<"\t"<<beta<<"\t"<<Cm<<"\t"<<targetV<<"\t"<<(*V)[idx]<<" "<<"\n";
#endif
      }
    }
  }
  else if (type == 3)
  {
    if (outFile)
    {
      if (getCurrentTime() > _time_for_io)
      {
        _time_for_io = getCurrentTime() + output_interval;
        (*outFile)<< std::fixed << getCurrentTime()<<"\t"<<
          targetV<<"\t"<<(*V)[idx]<<"\n";
      }
    }
  }
  else if (type == 2)
  {
    if (outFile)
    {
      if (getCurrentTime() > _time_for_io)
      {
        _time_for_io = getCurrentTime() + output_interval;
        (*outFile)<< std::fixed << getCurrentTime()<<"\t"<<
          targetV<<"\t"<<(*V)[idx]<<"\n";
      }
    }
  }
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

void VoltageClamp::setInjectedCurrent(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_VoltageClampInAttrPSet* CG_inAttrPset, CG_VoltageClampOutAttrPSet* CG_outAttrPset)
{
  idx = CG_inAttrPset->idx;
  if (idx < 0 or
      idx >= dimensions.size())  // if we pass in the InAttrPset with 'idx' attribute
  {//with a negative value
     // then inject at the last compartment (i.e. the closest to the soma)
    idx = dimensions.size()-1;
  }
}
