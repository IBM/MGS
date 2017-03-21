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

void VoltageClamp::initialize(RNG& rng) 
{
  outFile = new std::ofstream(fileName.c_str());
  (*outFile)<<"# Time\tCurrent\n";
  assert(deltaT);
  Vprev = (*V)[idx];
  waveformIdx = waveform.size();
}

void VoltageClamp::updateI(RNG& rng) 
{
  Igen = 0;
  bool inject=false;
  float targetV=0;
  if (isOn) {
    targetV=command;
    inject=true;
  }
  if (waveformIdx<waveform.size()) {
    targetV=waveform[waveformIdx];
    ++waveformIdx;
    inject=true;
  }
  if (inject) {
    Igen = beta * Cm * ( ( targetV - (*V)[idx] ) / *deltaT ) * *surfaceArea;
    (*outFile)<<getSimulation().getIteration()* *deltaT<<"\t"<<Igen<<" "<<Cm<<" "<<targetV<<" "<<(*V)[idx]<<" "<<*deltaT<<"\n";
  }
  Vprev = (*V)[idx];
}

void VoltageClamp::finalize(RNG& rng) 
{
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
    }
  }
}

void VoltageClamp::toggle(Trigger* trigger, NDPairList* ndPairList) 
{
  NDPairList::iterator iter=ndPairList->begin();
  NDPairList::iterator end=ndPairList->end();
  for (; iter!=end; ++iter) {
    if ( (*iter)->getName() == "toggle" ) {
      isOn = (static_cast<NumericDataItem*>((*iter)->getDataItem())->getInt()>0) ? true : false;
    }
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

