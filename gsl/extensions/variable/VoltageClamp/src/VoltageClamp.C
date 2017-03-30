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

#include "Lens.h"
#include "VoltageClamp.h"
#include "CG_VoltageClamp.h"
#include <memory>
#include <typeinfo>

void VoltageClamp::initialize(RNG& rng) 
{
  if (type == 1)
  {
    outFile = new std::ofstream(fileName.c_str());
    (*outFile)<<"# Time\tCurrent\n";
  }
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

  //NOTE: If not defined, idx=0 default
  surfaceArea = &(dimensions[idx]->surface_area);
  _isOn = false;
  _status = VoltageClamp::NO_CLAMP;

  _Vprev = (*V)[idx];
  waveformIdx = waveform.size();
}

void VoltageClamp::updateI(RNG& rng) 
{
  Igen = 0;
  bool inject=false;
  float targetV=0;
  if (_isOn) {
    targetV=command;
    inject=true;
  }
  if (waveformIdx<waveform.size()) {
    targetV=waveform[waveformIdx];
    ++waveformIdx;
    inject=true;
  }
  if (inject) {
    if (type == 1)
    {
      Igen = beta * Cm * ( ( targetV - (*V)[idx] ) / *deltaT ) * *surfaceArea;
      (*outFile)<<getSimulation().getIteration()* *deltaT<<"\t"<<Igen<<" "<<Cm<<" "<<targetV<<" "<<(*V)[idx]<<" "<<*deltaT<<"\n";
    }
    else if (type == 2)
    {
      (*V)[idx] = targetV;
      if (_status == VoltageClamp::SLOPE_ON)
      {
        float currentTime = getSimulation().getIteration() * (*deltaT);
        if (currentTime < _timeStart + gainTime)
        {
          (*V)[idx] = _Vstart + (currentTime - _timeStart)/(gainTime) * (targetV - _Vstart) ;
        }
        else{
          _status = VoltageClamp::FLAT_ZONE;
        }
      }
      else if (_status == VoltageClamp::SLOPE_OFF)
      {
        float currentTime = getSimulation().getIteration() * (*deltaT);
        if (currentTime < _timeStart + gainTime)
        {
          (*V)[idx] = targetV + (_timeStart + gainTime - currentTime )/(gainTime) * (_Vstart - targetV) ;
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
  if (type == 1)
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
      _timeStart = getSimulation().getIteration() * (*deltaT);
      _Vstart = (*V)[idx];
      if (_Vprev < command)
      {
        _status = VoltageClamp::SLOPE_ON;
      }
      else{
        _status = VoltageClamp::SLOPE_OFF;
      }
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
    _timeStart = getSimulation().getIteration() * (*deltaT);
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
  : CG_VoltageClamp()
{
}

VoltageClamp::~VoltageClamp() 
{
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

