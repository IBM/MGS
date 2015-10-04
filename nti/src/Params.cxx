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
// ================================================================

#include "Params.h"
#include "Segment.h"
#include "SegmentDescriptor.h"

#include <sstream>

Params::Params() :
  _bondK0(0), _bondR0(0), _nBondTypes(0),
  _angleK0(0), _angleR0(0), _nAngleTypes(0),
  _ljEps(0), _ljR0(0), _nLJTypes(0),
  _radiiMask(0), 
  _SIParamsMask(0),
  _compartmentVariableTargetsMask(0),
  _channelTargetsMask(0), 
  _electricalSynapseTargetsMask1(0), 
  _electricalSynapseTargetsMask2(0), 
  _chemicalSynapseTargetsMask1(0), 
  _chemicalSynapseTargetsMask2(0),
  _SIParams(false),
  _compartmentVariables(false),
  _channels(false), 
  _electricalSynapses(false), 
  _chemicalSynapses(false)
{
}

Params::Params(Params const & p) :
  _bondK0(0), _bondR0(0), _nBondTypes(p._nBondTypes),
  _angleK0(0), _angleR0(0), _nAngleTypes(p._nAngleTypes),
  _ljEps(0), _ljR0(0), _nLJTypes(p._nLJTypes), 
  _radiiMask(p._radiiMask), 
  _SIParamsMask(p._SIParamsMask), 
  _compartmentVariableTargetsMask(p._compartmentVariableTargetsMask),
  _channelTargetsMask(p._channelTargetsMask), 
  _electricalSynapseTargetsMask1(p._electricalSynapseTargetsMask1), 
  _electricalSynapseTargetsMask2(p._electricalSynapseTargetsMask2), 
  _chemicalSynapseTargetsMask1(p._chemicalSynapseTargetsMask1), 
  _chemicalSynapseTargetsMask2(p._chemicalSynapseTargetsMask2),
  _radiiMap(p._radiiMap),
  _SIParamsMap(p._SIParamsMap),
  _compartmentVariableTargetsMap(p._compartmentVariableTargetsMap),
  _channelTargetsMap(p._channelTargetsMap),
  _channelParamsMasks(p._channelParamsMasks),
  _channelParamsMap(p._channelParamsMap),
  _channelArrayParamsMap(p._channelArrayParamsMap),
  _compartmentParamsMasks(p._compartmentParamsMasks),
  _compartmentParamsMap(p._compartmentParamsMap),
  _compartmentArrayParamsMap(p._compartmentArrayParamsMap),
  _electricalSynapseTargetsMap(p._electricalSynapseTargetsMap),
  _chemicalSynapseTargetsMap(p._chemicalSynapseTargetsMap),
  _preSynapticPointTargetsMap(p._preSynapticPointTargetsMap),
  _preSynapticPointSynapseMap(p._preSynapticPointSynapseMap),
  _compartmentVariableCostsMap(p._compartmentVariableCostsMap),
  _channelCostsMap(p._channelCostsMap),
  _electricalSynapseCostsMap(p._electricalSynapseCostsMap),
  _chemicalSynapseCostsMap(p._chemicalSynapseCostsMap),
  _touchTableMasks(p._touchTableMasks),
  _SIParams(p._SIParams),
  _compartmentVariables(p._compartmentVariables),
  _channels(p._channels), 
  _electricalSynapses(p._electricalSynapses), 
  _chemicalSynapses(p._chemicalSynapses),
  _segmentDescriptor(p._segmentDescriptor)
{
  _bondK0 = new double[_nBondTypes];
  _bondR0 = new double[_nBondTypes];
  for(int i = 0 ; i < _nBondTypes; i++) {
    _bondK0[i]=p._bondK0[i];
    _bondR0[i]=p._bondR0[i];
  }
  _angleK0 = new double[_nAngleTypes];
  _angleR0 = new double[_nAngleTypes];
  for(int i = 0 ; i < _nAngleTypes; i++) {
    _angleK0[i]=p._angleK0[i];
    _angleR0[i]=p._angleR0[i];
  }
  _ljEps = new double[_nLJTypes];
  _ljR0 = new double[_nLJTypes];
  for(int i = 0 ; i < _nLJTypes; i++) {
    _ljEps[i]=p._ljEps[i];
    _ljR0[i]=p._ljR0[i];
  }
}

Params::Params(Params& p) :
  _bondK0(0), _bondR0(0), _nBondTypes(p._nBondTypes),
  _angleK0(0), _angleR0(0), _nAngleTypes(p._nAngleTypes),
  _ljEps(0), _ljR0(0), _nLJTypes(p._nLJTypes), 
  _radiiMask(p._radiiMask), 
  _SIParamsMask(p._SIParamsMask), 
  _compartmentVariableTargetsMask(p._compartmentVariableTargetsMask),
  _channelTargetsMask(p._channelTargetsMask), 
  _electricalSynapseTargetsMask1(p._electricalSynapseTargetsMask1), 
  _electricalSynapseTargetsMask2(p._electricalSynapseTargetsMask2), 
  _chemicalSynapseTargetsMask1(p._chemicalSynapseTargetsMask1), 
  _chemicalSynapseTargetsMask2(p._chemicalSynapseTargetsMask2),
  _radiiMap(p._radiiMap),
  _SIParamsMap(p._SIParamsMap),
  _compartmentVariableTargetsMap(p._compartmentVariableTargetsMap),
  _channelTargetsMap(p._channelTargetsMap),
  _channelParamsMasks(p._channelParamsMasks),
  _channelParamsMap(p._channelParamsMap),
  _channelArrayParamsMap(p._channelArrayParamsMap),
  _compartmentParamsMasks(p._compartmentParamsMasks),
  _compartmentParamsMap(p._compartmentParamsMap),
  _compartmentArrayParamsMap(p._compartmentArrayParamsMap),
  _electricalSynapseTargetsMap(p._electricalSynapseTargetsMap),
  _chemicalSynapseTargetsMap(p._chemicalSynapseTargetsMap),
  _preSynapticPointTargetsMap(p._preSynapticPointTargetsMap),
  _preSynapticPointSynapseMap(p._preSynapticPointSynapseMap),
  _compartmentVariableCostsMap(p._compartmentVariableCostsMap),
  _channelCostsMap(p._channelCostsMap),
  _electricalSynapseCostsMap(p._electricalSynapseCostsMap),
  _chemicalSynapseCostsMap(p._chemicalSynapseCostsMap),
  _touchTableMasks(p._touchTableMasks),
  _SIParams(p._SIParams),
  _compartmentVariables(p._compartmentVariables),
  _channels(p._channels), 
  _electricalSynapses(p._electricalSynapses), 
  _chemicalSynapses(p._chemicalSynapses),
  _segmentDescriptor(p._segmentDescriptor)
{
  _bondK0 = new double[_nBondTypes];
  _bondR0 = new double[_nBondTypes];
  for(int i = 0 ; i < _nBondTypes; i++) {
    _bondK0[i]=p._bondK0[i];
    _bondR0[i]=p._bondR0[i];
  }
  _angleK0 = new double[_nAngleTypes];
  _angleR0 = new double[_nAngleTypes];
  for(int i = 0 ; i < _nAngleTypes; i++) {
    _angleK0[i]=p._angleK0[i];
    _angleR0[i]=p._angleR0[i];
  }
  _ljEps = new double[_nLJTypes];
  _ljR0 = new double[_nLJTypes];
  for(int i = 0 ; i < _nLJTypes; i++) {
    _ljEps[i]=p._ljEps[i];
    _ljR0[i]=p._ljR0[i];
  }
}

Params::~Params()
{
  delete [] _bondR0;
  delete [] _bondK0;
  delete [] _angleR0;
  delete [] _angleK0;
  delete [] _ljR0;
  delete [] _ljEps;
}

void Params::readDevParams(const char *fname)
{
  FILE *fpF = fopen(fname,"r");
  assert(fpF);
  skipHeader(fpF);
  assert(readBondParams(fpF));
  //assert(readAngleParams(fpF));
  assert(readLJParams(fpF));
  readSIParams(fpF);
  readRadii(fpF);
  fclose(fpF);
}

void Params::readDetParams(const char *fname)
{
  FILE *fpF = fopen(fname,"r");
  assert(fpF);
  skipHeader(fpF);
  readRadii(fpF);
  readTouchTables(fpF);
  fclose(fpF);
}

void Params::readCptParams(const char *fname)
{
  FILE *fpF = fopen(fname,"r");
  assert(fpF);
  skipHeader(fpF);
  readCompartmentVariableTargets(fpF);
  readCompartmentVariableCosts(fpF);
  readModelParams(fpF, "COMPARTMENT_VARIABLE_PARAMS", _compartmentParamsMasks, _compartmentParamsMap, _compartmentArrayParamsMap);
  fclose(fpF);
}

void Params::readChanParams(const char *fname)
{
  FILE *fpF = fopen(fname,"r");
  assert(fpF);
  skipHeader(fpF);
  readChannelTargets(fpF);
  readChannelCosts(fpF);
  readModelParams(fpF, "CHANNEL_PARAMS", _channelParamsMasks, _channelParamsMap, _channelArrayParamsMap);
  fclose(fpF);
}

void Params::readSynParams(const char *fname)
{
  FILE *fpF = fopen(fname,"r");
  assert(fpF);
  skipHeader(fpF);
  readElectricalSynapseTargets(fpF);
  readElectricalSynapseCosts(fpF);
  readChemicalSynapseTargets(fpF);
  readChemicalSynapseCosts(fpF);
  readPreSynapticPointTargets(fpF);
  fclose(fpF);
}

bool Params::symmetricElectricalSynapseTargets(double key1, double key2)
{
  return (_segmentDescriptor.getSegmentKey(key1, _electricalSynapseTargetsMask1)==
	  _segmentDescriptor.getSegmentKey(key2, _electricalSynapseTargetsMask2));
}

double Params::getRadius(double key)
{
  double rval=0.0;
  std::map<double, double>::iterator iter=_radiiMap.find(_segmentDescriptor.getSegmentKey(key, _radiiMask));
  if (iter!=_radiiMap.end()) rval=iter->second;
  return rval;
}

SIParameters Params::getSIParams(double key1, double key2)
{
  SIParameters rval;
  if (_SIParams) {
    bool rvalSet=false;
    
    std::map<double, std::map<double, SIParameters> >::iterator iter1=
      _SIParamsMap.find(_segmentDescriptor.getSegmentKey(key1, _SIParamsMask));
    if (iter1!=_SIParamsMap.end()) {
      std::map<double, SIParameters>::iterator iter2=
	iter1->second.find(_segmentDescriptor.getSegmentKey(key2, _SIParamsMask));
      if (iter2!=iter1->second.end()) {
	rval=iter2->second;
	rvalSet=true;
      }
    }
  }
  return rval;
}

std::list<std::string> const * Params::getCompartmentVariableTargets(double key)
{
  std::list<std::string>* rval=0;
  if (_compartmentVariables) {
    std::map<double, std::list<std::string> >::iterator miter =
      _compartmentVariableTargetsMap.find(_segmentDescriptor.getSegmentKey(key, _compartmentVariableTargetsMask));
    if (miter!=_compartmentVariableTargetsMap.end()) {
      rval=&miter->second;
    }
  }
  return rval;
}

std::list<Params::ChannelTarget> * Params::getChannelTargets(double key)
{
  std::list<Params::ChannelTarget>* rval=0;
  if (_channels) {
    std::map<double, std::list<Params::ChannelTarget> >::iterator miter =
      _channelTargetsMap.find(_segmentDescriptor.getSegmentKey(key, _channelTargetsMask));
    if (miter!=_channelTargetsMap.end()) {
      rval=&miter->second;
    }
  }
  return rval;
}

std::list<Params::Params::ElectricalSynapseTarget> * Params::getElectricalSynapseTargets(double key1, double key2)
{
  std::list<Params::Params::ElectricalSynapseTarget>* rval=0;
  if (_electricalSynapses) {
    std::map<double, std::map<double, std::list<Params::Params::ElectricalSynapseTarget> > >::iterator miter1 =
      _electricalSynapseTargetsMap.find(_segmentDescriptor.getSegmentKey(key1, _electricalSynapseTargetsMask1));
    if (miter1!=_electricalSynapseTargetsMap.end()) {
      std::map<double, std::list<Params::Params::ElectricalSynapseTarget> >::iterator miter2 = 
	miter1->second.find(_segmentDescriptor.getSegmentKey(key2, _electricalSynapseTargetsMask2));
      if (miter2!=miter1->second.end()) {
	rval=&miter2->second;
      }
    }
  }
  return rval;
}

std::list<Params::ChemicalSynapseTarget> * Params::getChemicalSynapseTargets(double key1, double key2)
{
  std::list<Params::ChemicalSynapseTarget>* rval=0;
  if (_chemicalSynapses) {
    std::map<double, std::map<double, std::list<Params::ChemicalSynapseTarget> > >::iterator miter1 =
      _chemicalSynapseTargetsMap.find(_segmentDescriptor.getSegmentKey(key1, _chemicalSynapseTargetsMask1));
    if (miter1!=_chemicalSynapseTargetsMap.end()) {
      std::map<double, std::list<Params::ChemicalSynapseTarget> >::iterator miter2 = 
	miter1->second.find(_segmentDescriptor.getSegmentKey(key2, _chemicalSynapseTargetsMask2));
      if (miter2!=miter1->second.end()) {
	rval=&miter2->second;
      }
    }
  }
  return rval;
}

std::string Params::getPreSynapticPointTarget(std::string chemicalSynapseType)
{
  std::map<std::string, std::string>::iterator iter=_preSynapticPointTargetsMap.find(chemicalSynapseType);
  assert(iter!=_preSynapticPointTargetsMap.end());
  return (iter->second);
}

std::list<std::string>&  Params::getPreSynapticPointSynapseTypes(std::string targetType)
{
  std::map<std::string, std::list<std::string> >::iterator iter=_preSynapticPointSynapseMap.find(targetType);
  assert(iter!=_preSynapticPointSynapseMap.end());
  return iter->second;
}

bool Params::isCompartmentVariableTarget(double key, std::string type)
{
  bool rval=false;
  if (_compartmentVariables) {
    std::map<double, std::list<std::string> >::iterator miter =
      _compartmentVariableTargetsMap.find(_segmentDescriptor.getSegmentKey(key, _compartmentVariableTargetsMask));
    if (miter!=_compartmentVariableTargetsMap.end()) {
      std::list<std::string>::iterator titer=miter->second.begin(), tend=miter->second.end();
      for (; titer!=tend; ++titer) {
	if (*titer==type){
	  rval=true;
	  break;
	}
      }
    }
  }
  return rval;
}

bool Params::isChannelTarget(double key)
{
  bool rval=false;
  if (_channels) {
    std::map<double, std::list<Params::ChannelTarget> >::iterator miter =
      _channelTargetsMap.find(_segmentDescriptor.getSegmentKey(key, _channelTargetsMask));
    if (miter!=_channelTargetsMap.end()) {
      rval=true;
    }
  }
  return rval;
}

bool Params::isElectricalSynapseTarget(double key1, double key2, bool autapses)
{
  bool rval=false;
  for (int direction=0; direction<=1 && !rval; ++direction) {
    if ( _electricalSynapses &&
	 (key1<key2 || !symmetricElectricalSynapseTargets(key1, key2) ) &&
	 ( autapses ||
	   _segmentDescriptor.getNeuronIndex(key1)!= 
	   _segmentDescriptor.getNeuronIndex(key2) ) ) {
      std::map<double, std::map<double, std::list<Params::ElectricalSynapseTarget> > >::iterator miter=
	_electricalSynapseTargetsMap.find(_segmentDescriptor.getSegmentKey(key1, _electricalSynapseTargetsMask1) );
      if (miter!=_electricalSynapseTargetsMap.end() ) {
	std::map<double, std::list<Params::ElectricalSynapseTarget> >::iterator miter2=
	  miter->second.find(_segmentDescriptor.getSegmentKey(key2, _electricalSynapseTargetsMask2) );
	if (miter2!=miter->second.end()) {
	  rval=true;
	}
      }
    }
    double tmp=key1;
    key1=key2;
    key2=tmp;
  }
  return rval;
}

bool Params::isElectricalSynapseTarget(double key)
{
  bool rval=false;
  if (_electricalSynapses) {
    std::map<double, std::map<double, std::list<Params::ElectricalSynapseTarget> > >::const_iterator miter=_electricalSynapseTargetsMap.begin(),
      mend=_electricalSynapseTargetsMap.end();
    rval=(_electricalSynapseTargetsMap.find(_segmentDescriptor.getSegmentKey(key, _electricalSynapseTargetsMask1) ) != mend || 
	  _electricalSynapseTargetsMap.find(_segmentDescriptor.getSegmentKey(key, _electricalSynapseTargetsMask2) ) != mend);
    for (; miter!=mend && !rval; ++miter) {
      std::map<double, std::list<Params::ElectricalSynapseTarget> >::const_iterator mend2=miter->second.end();
      rval=(miter->second.find(_segmentDescriptor.getSegmentKey(key, _electricalSynapseTargetsMask1) ) != mend2 || 
	    miter->second.find(_segmentDescriptor.getSegmentKey(key, _electricalSynapseTargetsMask2) ) != mend2);	
    }
  }
  return rval;
}

bool Params::isChemicalSynapseTarget(double key1, double key2, bool autapses)
{
  bool rval=false;
  //for (int direction=0; direction<=1 && !rval; ++direction) {
    if ( _chemicalSynapses &&
	 ( autapses ||
	   _segmentDescriptor.getNeuronIndex(key1)!= 
	   _segmentDescriptor.getNeuronIndex(key2) ) ) {
      std::map<double, std::map<double, std::list<Params::ChemicalSynapseTarget> > >::iterator miter=
	_chemicalSynapseTargetsMap.find(_segmentDescriptor.getSegmentKey(key1, _chemicalSynapseTargetsMask1) );
      if (miter!=_chemicalSynapseTargetsMap.end() ) {
	std::map<double, std::list<Params::ChemicalSynapseTarget> >::iterator miter2=
	  miter->second.find(_segmentDescriptor.getSegmentKey(key2, _chemicalSynapseTargetsMask2) );
	if (miter2!=miter->second.end()) {
	  rval=true;
	}
      }
    }
    /*
    double tmp=key1;
    key1=key2;
    key2=tmp;
  }
    */
  return rval;
}

bool Params::isChemicalSynapseTarget(double key)
{
  bool rval=false;
  if (_chemicalSynapses) {
    std::map<double, std::map<double, std::list<Params::ChemicalSynapseTarget> > >::const_iterator miter=_chemicalSynapseTargetsMap.begin(),
      mend=_chemicalSynapseTargetsMap.end();
    rval=(_chemicalSynapseTargetsMap.find(_segmentDescriptor.getSegmentKey(key, _chemicalSynapseTargetsMask1) ) != mend || 
	  _chemicalSynapseTargetsMap.find(_segmentDescriptor.getSegmentKey(key, _chemicalSynapseTargetsMask2) ) != mend);
    for (; miter!=mend && !rval; ++miter) {
      std::map<double, std::list<Params::ChemicalSynapseTarget> >::const_iterator mend2=miter->second.end();
      rval=(miter->second.find(_segmentDescriptor.getSegmentKey(key, _chemicalSynapseTargetsMask1) ) != mend2 || 
	    miter->second.find(_segmentDescriptor.getSegmentKey(key, _chemicalSynapseTargetsMask2) ) != mend2);	
    }
  }
  return rval;
}

double Params::getCompartmentVariableCost(std::string compartmentVariableId)
{
  double rval=0.0;
  std::map<std::string, double>::iterator miter =
    _compartmentVariableCostsMap.find(compartmentVariableId);
  if (miter!=_compartmentVariableCostsMap.end()) {
    rval=miter->second;
  }
  else {
    std::cerr<<"Params : Unspecified CompartmentVariable Cost! ID : "
	     <<compartmentVariableId<<std::endl;
    exit(0);
  }
  return rval;
}

double Params::getChannelCost(std::string channelId)
{
  double rval=0.0;
  std::map<std::string, double>::iterator miter =
    _channelCostsMap.find(channelId);
  if (miter!=_channelCostsMap.end()) {
    rval=miter->second;
  }
  else {
    std::cerr<<"Params : Unspecified Channel Cost! ID : "
	     <<channelId<<std::endl;
    exit(0);
  }
  return rval;
}

double Params::getElectricalSynapseCost(std::string electricalSynapseId)
{
  double rval=0.0;
  std::map<std::string, double>::iterator miter =
    _electricalSynapseCostsMap.find(electricalSynapseId);
  if (miter!=_electricalSynapseCostsMap.end()) {
    rval=miter->second;
  }
  else {
    std::cerr<<"Params : Unspecified Electrical Synapse Cost! ID : "
	     <<electricalSynapseId<<std::endl;
    exit(0);
  }
  return rval;
}

double Params::getChemicalSynapseCost(std::string chemicalSynapseId)
{
  double rval=0.0;
  std::map<std::string, double>::iterator miter =
    _chemicalSynapseCostsMap.find(chemicalSynapseId);
  if (miter!=_chemicalSynapseCostsMap.end()) {
    rval=miter->second;
  }
  else {
    std::cerr<<"Params : Unspecified Chemical Synapse Cost! ID: "
	     <<chemicalSynapseId<<std::endl;
    exit(0);
  }
  return rval;
}

void Params::getModelParams(ModelType modelType, std::string nodeType, double key, std::list<std::pair<std::string, float> >& modelParams)
{
  std::map<std::string, unsigned long long>* modelParamsMasks;
  std::map<std::string, std::map<double, std::list<std::pair<std::string, float> > > >* modelParamsMap;

  switch (modelType) {
  case COMPARTMENT :
    modelParamsMasks=&_compartmentParamsMasks;
    modelParamsMap=&_compartmentParamsMap;
    break;
  case CHANNEL :
    modelParamsMasks=&_channelParamsMasks;
    modelParamsMap=&_channelParamsMap;
    break;
  }

  modelParams.clear();
  std::map<std::string, std::map<double, std::list<std::pair<std::string, float> > > >::iterator iter1=modelParamsMap->find(nodeType);
  if (iter1!=modelParamsMap->end()) {
    std::map<std::string, unsigned long long>::iterator miter=modelParamsMasks->find(nodeType);
    assert(miter!=modelParamsMasks->end());
    std::map<double, std::list<std::pair<std::string, float> > >::iterator iter2=(iter1->second).find(_segmentDescriptor.getSegmentKey(key, miter->second) );
    if (iter2!=(iter1->second).end()) {
      modelParams=iter2->second;
    }
  }
}

void Params::getModelArrayParams(ModelType modelType, std::string nodeType, double key, std::list<std::pair<std::string, std::vector<float> > >& modelArrayParams)
{
  std::map<std::string, unsigned long long>* modelParamsMasks;
  std::map<std::string, std::map<double, std::list<std::pair<std::string, std::vector<float> > > > >* modelArrayParamsMap;

  switch (modelType) {
  case COMPARTMENT :
    modelParamsMasks=&_compartmentParamsMasks;
    modelArrayParamsMap=&_compartmentArrayParamsMap;
    break;
  case CHANNEL :
    modelParamsMasks=&_channelParamsMasks;
    modelArrayParamsMap=&_channelArrayParamsMap;
    break;
  }

  modelArrayParams.clear();
  std::map<std::string, std::map<double, std::list<std::pair<std::string, std::vector<float> > > > >::iterator iter1=modelArrayParamsMap->find(nodeType);
  if (iter1!=modelArrayParamsMap->end()) {
    std::map<std::string, unsigned long long>::iterator miter=modelParamsMasks->find(nodeType);
    assert(miter!=modelParamsMasks->end());
    std::map<double, std::list<std::pair<std::string, std::vector<float> > > >::iterator iter2=(iter1->second).find(_segmentDescriptor.getSegmentKey(key, miter->second) );
    if (iter2!=(iter1->second).end()) {
      modelArrayParams=iter2->second;
    }
  }
}

bool Params::readBondParams(FILE *fpF)
{
  bool rval=false;
  int n=_nBondTypes=0;
  char bufS[1024], tokS[256];
  char* c=fgets(bufS,1024,fpF);
  while(!feof(fpF)) {
    if(2 == sscanf(bufS,"%s %d ", tokS, &n)) {
      if(!strcmp("NBONDTYPES", tokS)) {
	_nBondTypes=n;
	break;
      }
    }
    c=fgets(bufS,1024,fpF);
  }
  delete [] _bondR0;
  delete [] _bondK0;
  _bondR0=_bondK0=0;
  if (_nBondTypes>0) {
    _bondR0 = new double[_nBondTypes];
    _bondK0 = new double[_nBondTypes];
    for(int i=0 ; i<_nBondTypes; i++) {
      c=fgets(bufS,1024,fpF);
      if (2!=sscanf(bufS,"%lf %lf ", &_bondK0[i], &_bondR0[i])) assert(0);
    }
    rval=true;
  }
  return rval;
}

bool Params::readAngleParams(FILE *fpF)
{
  bool rval=false;
  int n=_nAngleTypes=0;
  char bufS[1024], tokS[256];
  char* c=fgets(bufS,1024,fpF);
  while(!feof(fpF)) {
    if(2 == sscanf(bufS,"%s %d ", tokS, &n)) {
      if(!strcmp("NANGLETYPES", tokS)) { 
	_nAngleTypes=n; 
	break;
      }
    }
    c=fgets(bufS,1024,fpF);
  }
  delete [] _angleR0;
  delete [] _angleK0;
  _angleR0=_angleK0=0;
  if (_nAngleTypes>0) {
    _angleR0 = new double[_nAngleTypes];
    _angleK0 = new double[_nAngleTypes];
    for(int i=0; i<_nAngleTypes; i++) {
      c=fgets(bufS,1024,fpF);
      if (2!=sscanf(bufS,"%lf %lf ", &_angleK0[i], &_angleR0[i])) assert(0);
      _angleR0[i] *= (M_PI/180.0);
    }
    rval=true;
  }
  return rval;
}

bool Params::readLJParams(FILE *fpF)
{
  bool rval=false;
  int n=_nLJTypes=0;
  char bufS[1024], tokS[256];
  char* c=fgets(bufS,1024,fpF);
  while(!feof(fpF)) {
    if(2 == sscanf(bufS,"%s %d ", tokS, &n)) {
      if( !strcmp("NLJTYPES", tokS) ||  !strcmp("NREPULSETYPES", tokS)) {
	_nLJTypes=n;
	break;
      }
    }
    c=fgets(bufS,1024,fpF);
  }
  delete [] _ljR0;
  delete [] _ljEps;
  _ljR0=_ljEps=0;
  if (_nLJTypes>0) {
    _ljR0 = new double[_nLJTypes];
    _ljEps = new double[_nLJTypes];
    for(int i=0; i<_nLJTypes; i++) {
      c=fgets(bufS,1024,fpF);
      if (2!=sscanf(bufS,"%lf %lf ", &_ljEps[i], &_ljR0[i])) assert(0);
      _ljEps[i] = sqrt(_ljEps[i]);
    }
    rval=true;
  }
  return rval;
}

bool Params::readRadii(FILE *fpF)
{
  bool rval=false;
  _radiiMask=0;
  _radiiMap.clear();
  int n=0;
  char bufS[1024], tokS[256];
  char* c=fgets(bufS,1024,fpF);
  while(!feof(fpF)) {
    if(2 == sscanf(bufS,"%s %d ", tokS, &n)) {
      if(!strcmp("RADII", tokS)) break;
    }
    c=fgets(bufS,1024,fpF);
  }
  if (n>0) {
    std::vector<SegmentDescriptor::SegmentKeyData> maskVector;
    _radiiMask = resetMask(fpF, maskVector);
    double radius;
    unsigned int sz=maskVector.size();
    assert(sz);
    unsigned int* ids = new unsigned int[sz];

    for(int i=0; i<n; i++) {
      for (int j=0; j<sz; ++j) {
	if (1!=fscanf(fpF, "%d", &ids[j])) assert(0);
      }
      c=fgets(bufS,1024,fpF);
      if (1!=sscanf(bufS,"%lf", &radius)) assert(0);
      _radiiMap[_segmentDescriptor.getSegmentKey(maskVector, &ids[0])]=radius;
    }
    delete [] ids;
    rval=true;
  }
  return rval;
}

bool Params::readTouchTables(FILE *fpF)
{
  bool rval=false;
  _touchTableMasks.clear();
  int n=0;
  char bufS[1024], tokS[256];
  char* c=fgets(bufS,1024,fpF);
  while(!feof(fpF)) {
    if(2 == sscanf(bufS,"%s %d ", tokS, &n)) {
      if(!strcmp("TOUCH_TABLES", tokS)) break;
    }
    c=fgets(bufS,1024,fpF);
  }
  for(int i=0; i<n; i++) {
    std::vector<SegmentDescriptor::SegmentKeyData> maskVector;
    resetMask(fpF, maskVector);
    _touchTableMasks.push_back(maskVector);
    rval=true;
  }
  return rval;
}

bool Params::readSIParams(FILE *fpF)
{
  _SIParams=false;
  _SIParamsMask=0;
  _SIParamsMap.clear();
  int n=0;
  char bufS[1024], tokS[256];
  char* c=fgets(bufS,1024,fpF);
  while(!feof(fpF)) {
    if(2 == sscanf(bufS,"%s %d ", tokS, &n)) {
      if(!strcmp("NSITYPES", tokS)) {
	break;
      }
    }
    char* c=fgets(bufS,1024,fpF);
  }
  if (n>0) {
    std::vector<SegmentDescriptor::SegmentKeyData> maskVector;
    _SIParamsMask = resetMask(fpF, maskVector);

    double Epsilon, Sigma;
    unsigned int sz=maskVector.size()*2;
    unsigned int* ids = new unsigned int[sz];

    for(int i=0; i<n; i++) {
      for (int j=0; j<sz; ++j) {
	if (1!=fscanf(fpF, "%d", &ids[j])) assert(0);
      }
      c=fgets(bufS,1024,fpF);
      if (2!=sscanf(bufS,"%lf %lf ", &Epsilon, &Sigma)) assert(0);
      double key1=_segmentDescriptor.getSegmentKey(maskVector, &ids[0]);
      double key2=_segmentDescriptor.getSegmentKey(maskVector, &ids[sz/2]);
      std::map<double, std::map<double, SIParameters> >::iterator 
	iter=_SIParamsMap.find(key1);
      if (iter==_SIParamsMap.end()) {
	std::map<double, SIParameters> newMap;
	(newMap[key2]).Epsilon=Epsilon;
	(newMap[key2]).Sigma=Sigma;
	_SIParamsMap[key1]=newMap;
      }
      else {
	((*iter).second)[key2].Epsilon=Epsilon;
	((*iter).second)[key2].Sigma=Sigma;
      }
    }
    delete [] ids;
    _SIParams=true;
  }
  return _SIParams;
}

bool Params::readCompartmentVariableTargets(FILE* fpF)
{
  _compartmentVariables=false;
  _compartmentVariableTargetsMask=0;
  _compartmentVariableTargetsMap.clear();
  skipHeader(fpF);
  int n=0;
  char bufS[1024], tokS[256];
  char* c=fgets(bufS,1024,fpF);
  while(!feof(fpF)) {
    if(2 == sscanf(bufS,"%s %d ", tokS, &n)) {
      if(!strcmp("COMPARTMENT_VARIABLE_TARGETS", tokS)) break;
    }
    c=fgets(bufS,1024,fpF);
  }
  if (n>0) {
    std::vector<SegmentDescriptor::SegmentKeyData> maskVector;
    _compartmentVariableTargetsMask=resetMask(fpF, maskVector);
    unsigned int sz=maskVector.size();
    assert(sz);
    unsigned int* ids = new unsigned int[sz];
    
    for(int i=0; i<n; i++) {
      for (int j=0; j<sz; ++j) {
	if (1!=fscanf(fpF, "%d", &ids[j])) assert(0);
	if (maskVector[j]==SegmentDescriptor::segmentIndex) {
	  std::cerr<<"Params : Targeting compartmentVariables to individual compartments not supported!"<<std::endl;
	  exit(0);
	}
      }
      assert(!feof(fpF));
      c=fgets(bufS,1024,fpF);
      std::istringstream is(bufS);
      std::list<std::string> targets;
      std::string type;
      while(is >> type) {
	targets.push_back(type);
      }
      targets.sort();
      _compartmentVariableTargetsMap[_segmentDescriptor.getSegmentKey(maskVector, &ids[0])]=targets;
    }
    delete [] ids;
    _compartmentVariables=true;
  }
  return _compartmentVariables;
}

bool Params::readChannelTargets(FILE *fpF)
{
  _channels=false;
  _channelTargetsMask=0;
  _channelTargetsMap.clear();
  skipHeader(fpF);
  int n=0;
  char bufS[1024], tokS[256];
  char* c=fgets(bufS,1024,fpF);
  while(!feof(fpF)) {
    if(2 == sscanf(bufS,"%s %d ", tokS, &n)) {
      if(!strcmp("CHANNEL_TARGETS", tokS)) break;
    }
    c=fgets(bufS,1024,fpF);
  }
  if (n>0) {
    std::vector<SegmentDescriptor::SegmentKeyData> maskVector;
    _channelTargetsMask=resetMask(fpF, maskVector);
    unsigned int sz=maskVector.size();
    assert(sz);
    unsigned int* ids = new unsigned int[sz];
    
    for(int i=0; i<n; i++) {
      for (int j=0; j<sz; ++j) {
	if (1!=fscanf(fpF, "%d", &ids[j])) assert(0);
	if (maskVector[j]==SegmentDescriptor::segmentIndex) {
	  std::cerr<<"Params : Targeting channels to individual compartments not supported!"<<std::endl;
	  exit(0);
	}
      }
      assert(!feof(fpF));
      c=fgets(bufS,1024,fpF);
      std::istringstream is(bufS);
      std::list<Params::ChannelTarget>& targets=_channelTargetsMap[_segmentDescriptor.getSegmentKey(maskVector, &ids[0])];      
      Params::ChannelTarget ct;
      while(is >> ct._type) {
	while (is.get()!='[') {assert(is.good());}
	char buf1[256];
	is.get(buf1, 256, ']');
	char* tok1=strtok(buf1, " ,");
	while (tok1!=0) {
	  ct.addTarget1(std::string(tok1));
	  tok1=strtok(0, " ,");
	}
	if (is.get()!=']') assert(0);
	while (is.get()!='[') {assert(is.good());}
	char buf2[256];
	is.get(buf2, 256, ']');
	if (is.get()!=']') assert(0);
	char* tok2=strtok(buf2, " ,");
	while (tok2!=0) {
	  ct.addTarget2(std::string(tok2));
	  tok2=strtok(0, " ,");
	}
	targets.push_back(ct);
	ct.clear();
      }
      targets.sort();
    }
    delete [] ids;
    _channels=true;
  }
  return _channels;
}

bool Params::readElectricalSynapseTargets(FILE* fpF)
{
  _electricalSynapses=false;
  _electricalSynapseTargetsMask1=_electricalSynapseTargetsMask2=0;
  _electricalSynapseTargetsMap.clear();
  skipHeader(fpF);
  int n=0;
  char bufS[1024];
  std::string tokS;
  char* c=fgets(bufS,1024,fpF);
  while(!feof(fpF)) {
    std::istringstream is(bufS);
    is>>tokS;
    if(tokS=="ELECTRICAL_SYNAPSE_TARGETS") {
      is>>n;
      break;
    }
    c=fgets(bufS,1024,fpF);
  }
  if (n>0) {
    std::vector<SegmentDescriptor::SegmentKeyData> maskVector1, maskVector2;
    _electricalSynapseTargetsMask1=resetMask(fpF, maskVector1);
    unsigned int sz1=maskVector1.size();
    assert(sz1);
    _electricalSynapseTargetsMask2=resetMask(fpF, maskVector2);
    unsigned int sz2=maskVector2.size();
    assert(sz2);

    unsigned int* ids1 = new unsigned int[sz1];
    unsigned int* ids2 = new unsigned int[sz2];

    for(int i=0 ; i<n; ++i) {
      for (int j=0; j<sz1; ++j) {
	if (1!=fscanf(fpF, "%d", &ids1[j])) assert(0);
      }
      for (int j=0; j<sz2; ++j) {
	if (1!=fscanf(fpF, "%d", &ids2[j])) assert(0);
      }
      assert(!feof(fpF));
      c=fgets(bufS,1024,fpF);
      std::istringstream is(bufS);

      std::map<double, std::list<Params::ElectricalSynapseTarget> >& targetsMap = 
	_electricalSynapseTargetsMap[_segmentDescriptor.getSegmentKey(maskVector1, &ids1[0])];
      std::list<Params::ElectricalSynapseTarget>& targets =
	targetsMap[_segmentDescriptor.getSegmentKey(maskVector2, &ids2[0])];

      Params::ElectricalSynapseTarget st;
      st._parameter=-1.0;
      while (is>>st._type) {
	while (is.get()!='[') {assert(is.good());}
	char buf[256];
	is.get(buf, 256, ']');
	char* tok=strtok(buf, " ,");
	while (tok!=0) {
	  st.addTarget(std::string(tok));
	  tok=strtok(0, " ,");
	}
	if (is.get()!=']') assert(0);
	is>>st._parameter;
	targets.push_back(st);
	st.clear();
      }
      targets.sort();
    }
    delete [] ids1;
    delete [] ids2;
    _electricalSynapses=true;
  }
   return _electricalSynapses;
}

bool Params::readChemicalSynapseTargets(FILE* fpF)
{
  _chemicalSynapses=false;
  _chemicalSynapseTargetsMask1=_chemicalSynapseTargetsMask2=0;
  _chemicalSynapseTargetsMap.clear();
  skipHeader(fpF);
  int n=0;
  char bufS[1024];
  std::string tokS;
  char* c=fgets(bufS,1024,fpF);
  while(!feof(fpF)) {
    std::istringstream is(bufS);
    is>>tokS;
    if(tokS=="CHEMICAL_SYNAPSE_TARGETS") {
      is>>n;
      break;
    }
    c=fgets(bufS,1024,fpF);
  }
  if (n>0) {
    std::vector<SegmentDescriptor::SegmentKeyData> maskVector1, maskVector2;
    _chemicalSynapseTargetsMask1=resetMask(fpF, maskVector1);
    unsigned int sz1=maskVector1.size();
    assert(sz1);
    _chemicalSynapseTargetsMask2=resetMask(fpF, maskVector2);
    unsigned int sz2=maskVector2.size();
    assert(sz2);

    unsigned int* ids1 = new unsigned int[sz1];
    unsigned int* ids2 = new unsigned int[sz2];

    for(int i=0; i<n; ++i) {
      for (int j=0; j<sz1; ++j) {
	if (1!=fscanf(fpF, "%d", &ids1[j])) assert(0);
      }
      for (int j=0; j<sz2; ++j) {
	if (1!=fscanf(fpF, "%d", &ids2[j])) assert(0);
      }
      assert(!feof(fpF));
      c=fgets(bufS,1024,fpF);
      std::istringstream is(bufS);

      std::map<double, std::list<Params::ChemicalSynapseTarget> >& targetsMap = 
	_chemicalSynapseTargetsMap[_segmentDescriptor.getSegmentKey(maskVector1, &ids1[0])];
      std::list<Params::ChemicalSynapseTarget>& targets =
	targetsMap[_segmentDescriptor.getSegmentKey(maskVector2, &ids2[0])];

      Params::ChemicalSynapseTarget st;
      st._parameter=-1.0;

      std::vector<std::string> types;

      while (is.get()!='[') {assert(is.good());}
      char buf1[256];
      is.get(buf1, 256, ']');
      char* tok=strtok(buf1, " ,");
      while (tok!=0) {
	types.push_back(std::string(tok));
	tok=strtok(0, " ,");
      }
      if (is.get()!=']') assert(0);

      for (int i=0; i<types.size(); ++i) {
	while (is.get()!='[') {assert(is.good());}
	char buf1[256];
	is.get(buf1, 256, ']');
	char* tok1=strtok(buf1, " ,");
	while (tok1!=0) {
	  st.addTarget1(types[i], std::string(tok1));
	  tok1=strtok(0, " ,");
	}
	if (is.get()!=']') assert(0);
	while (is.get()!='[') {assert(is.good());}
	char buf2[256];
	is.get(buf2, 256, ']');
	if (is.get()!=']') assert(0);
	char* tok2=strtok(buf2, " ,");
	while (tok2!=0) {
	  st.addTarget2(types[i], std::string(tok2));
	  tok2=strtok(0, " ,");
	}
      }
      is>>st._parameter;
      targets.push_back(st);
      st.clear();
      targets.sort();
    }
    delete [] ids1;
    delete [] ids2;
    _chemicalSynapses=true;
  }
  return _chemicalSynapses;
}

bool Params::readPreSynapticPointTargets(FILE* fpF)
{
  bool rval=false;
  _preSynapticPointTargetsMap.clear();
  _preSynapticPointSynapseMap.clear();
  skipHeader(fpF);
  int n=0;
  char bufS[1024], tokS[256], tokS2[256];
  char* c=fgets(bufS,1024,fpF);
  while(!feof(fpF)) {
    if(2 == sscanf(bufS,"%s %d ", tokS, &n)) {
      if(!strcmp("PRESYNAPTIC_POINT_TARGETS", tokS)) break;
    }
    c=fgets(bufS,1024,fpF);
  }
  if (n>0) {
    for(int i = 0 ; i < n; i++) {
      c=fgets(bufS,1024,fpF);
      if(2 == sscanf(bufS,"%s %s ", tokS, tokS2)) {
	std::string synID(tokS);
	std::string targetID(tokS2);
	_preSynapticPointTargetsMap[synID]=targetID;
	_preSynapticPointSynapseMap[targetID].push_back(synID);
      }
      else assert(0);
    }
    rval=true;
  }
  return rval;
}

void Params::skipHeader(FILE* fpF)
{
  int pos=ftell(fpF);
  char bufS[1024];
  do {
    pos=ftell(fpF);
    char* c=fgets(bufS,1024,fpF);
  } while (bufS[0]=='#');
  fseek(fpF, pos, SEEK_SET);
}

unsigned long long Params::readNamedParam(FILE *fpF, std::string name, std::map<double, double>& namedParamsMap)
{
  unsigned long long mask=0;
  namedParamsMap.clear();
  int n=0;
  char bufS[1024], tokS[256];
  char* c=fgets(bufS,1024,fpF);
  while(!feof(fpF)) {
    if(2 == sscanf(bufS,"%s %d ", tokS, &n)) {
      if(!strcmp(name.c_str(), tokS)) break;
    }
    c=fgets(bufS,1024,fpF);
  }
  assert(n>0);
  
  std::vector<SegmentDescriptor::SegmentKeyData> maskVector;
  mask = resetMask(fpF, maskVector);
  double p;
  unsigned int sz=maskVector.size();
  assert(sz);
  unsigned int* ids = new unsigned int[sz];

  for(int i = 0 ; i < n; i++) {
    for (int j=0; j<sz; ++j) {
      if (1!=fscanf(fpF, "%d", &ids[j])) assert(0);
    }
    assert(!feof(fpF));
    c=fgets(bufS,1024,fpF);
    sscanf(bufS,"%lf", &p);
    namedParamsMap[_segmentDescriptor.getSegmentKey(maskVector, &ids[0])]=p;
  }
  delete [] ids;
  return mask;
}

bool Params::readCompartmentVariableCosts(FILE* fpF)
{
  bool rval=false;
  _compartmentVariableCostsMap.clear();
  skipHeader(fpF);
  int n=0;
  char bufS[1024], tokS[256];
  char* c=fgets(bufS,1024,fpF);
  while(!feof(fpF)) {
    if(2 == sscanf(bufS,"%s %d ", tokS, &n)) {
      if(!strcmp("COMPARTMENT_VARIABLE_COSTS", tokS)) break;
    }
    c=fgets(bufS,1024,fpF);
  }
  if (n>0) {
    double cost;
    for(int i = 0 ; i < n; i++) {
      c=fgets(bufS,1024,fpF);
      if(2 == sscanf(bufS,"%s %lf ", tokS, &cost)) {
	std::string chanID(tokS);
	_compartmentVariableCostsMap[chanID]=cost;
      }
      else assert(0);
    }
    rval=true;
  }
  return rval;
}

bool Params::readChannelCosts(FILE* fpF)
{
  bool rval=false;
  _channelCostsMap.clear();
  skipHeader(fpF);
  int n=0;
  char bufS[1024], tokS[256];
  char* c=fgets(bufS,1024,fpF);
  while(!feof(fpF)) {
    if(2 == sscanf(bufS,"%s %d ", tokS, &n)) {
      if(!strcmp("CHANNEL_COSTS", tokS)) break;
    }
    c=fgets(bufS,1024,fpF);
  }
  if (n>0) {
    double cost;
    for(int i = 0 ; i < n; i++) {
      c=fgets(bufS,1024,fpF);
      if(2 == sscanf(bufS,"%s %lf ", tokS, &cost)) {
	std::string chanID(tokS);
	_channelCostsMap[chanID]=cost;
      }
      else assert(0);
    }
    rval=true;
  }
  return rval;
}

bool Params::readModelParams(FILE* fpF, const char* id, 
			     std::map<std::string, unsigned long long>& paramsMasks,
			     std::map<std::string, std::map<double, std::list<std::pair<std::string, float> > > >& paramsMap,
			     std::map<std::string, std::map<double, std::list<std::pair<std::string, std::vector<float> > > > >& arrayParamsMap)
{
  bool rval=false;
  paramsMasks.clear();
  paramsMap.clear();
  skipHeader(fpF);
  int n=0;
  char bufS[1024], tokS[256];
  char* c=fgets(bufS,1024,fpF);
  while(!feof(fpF)) {
    if(2 == sscanf(bufS,"%s %d ", tokS, &n)) {
      if(!strcmp(id, tokS)) break;
    }
    c=fgets(bufS,1024,fpF);
  }
  if (n>0) {
    for(int i = 0; i < n; i++) {
      int p;
      c=fgets(bufS,1024,fpF);
      if(2 == sscanf(bufS,"%s %d ", tokS, &p)) {
	std::string modelID(tokS);
	std::vector<SegmentDescriptor::SegmentKeyData> maskVector;
	paramsMasks[modelID]=resetMask(fpF, maskVector);
	unsigned int sz=maskVector.size();
	assert(sz);
	unsigned int* ids = new unsigned int[sz];
	for (int j=0; j<p; ++j) {
	  for (int k=0; k<sz; ++k) {
	    if (1!=fscanf(fpF, "%d", &ids[k])) assert(0);
	    if (maskVector[k]==SegmentDescriptor::segmentIndex) {
	      std::cerr<<"Params : Targeting channel parameters to individual compartments not supported!"<<std::endl;
	      exit(0);
	    }
	  }
	  assert(!feof(fpF));
	  c=fgets(bufS,1024,fpF);
	  std::istringstream is(bufS);
	  std::list<std::pair<std::string, float> >& params=paramsMap[modelID][_segmentDescriptor.getSegmentKey(maskVector, &ids[0])];
	  std::list<std::pair<std::string, std::vector<float> > >& arrayParams=arrayParamsMap[modelID][_segmentDescriptor.getSegmentKey(maskVector, &ids[0])];
	  while (is.get()!='<') {assert(is.good());}
	  char buf1[256];
	  is.get(buf1, 256, '>');
	  char* tok1=strtok(buf1, ";");
	  while (tok1!=0) {
	    char* tok2=strtok(tok1, "=");
	    std::string name(tok2);
	    tok2=strtok(0, " =");
	    std::istringstream is2(tok2);
	    if (is2.get()!='{') {
	      float value=atof(tok2);
	      params.push_back(std::pair<std::string, float>(name, value) );
	    }
	    else {
	      std::vector<float> value;
	      char buf2[256];
	      is2.get(buf2, 256, '}');
	      char* tok3=strtok(buf2, ",");
	      while (tok3!=0) {
		value.push_back(atof(tok3));
		tok3=strtok(0, ",");
	      }
	      arrayParams.push_back(std::pair<std::string, std::vector<float> >(name,value) );
	    }
	    tok1=strtok(0, ";");
	  }
	}
	delete [] ids;
      }
      else assert(0);
    }
    rval=true;
  }
  return rval;
}

bool Params::readElectricalSynapseCosts(FILE* fpF)
{
  bool rval=false;
  _electricalSynapseCostsMap.clear();
  skipHeader(fpF);
  int n=0;
  char bufS[1024], tokS[256];
  char* c=fgets(bufS,1024,fpF);
  while(!feof(fpF)) {
    if(2 == sscanf(bufS,"%s %d ", tokS, &n)) {
      if(!strcmp("ELECTRICAL_SYNAPSE_COSTS", tokS)) break;
    }
    c=fgets(bufS,1024,fpF);
  }
  if (n>0) {
    double cost;
    for(int i = 0 ; i < n; i++) {
      c=fgets(bufS,1024,fpF);
      if(2 == sscanf(bufS,"%s %lf ", tokS, &cost)) {
	std::string synID(tokS);
	_electricalSynapseCostsMap[synID]=cost;
      }
      else assert(0);
    }
    rval=true;
  }
  return rval;
}


bool Params::readChemicalSynapseCosts(FILE* fpF)
{
  bool rval=false;
  _chemicalSynapseCostsMap.clear();
  skipHeader(fpF);
  int n=0;
  char bufS[1024], tokS[256];
  char* c=fgets(bufS,1024,fpF);
  while(!feof(fpF)) {
    if(2 == sscanf(bufS,"%s %d ", tokS, &n)) {
      if(!strcmp("CHEMICAL_SYNAPSE_COSTS", tokS)) break;
    }
    c=fgets(bufS,1024,fpF);
  }
  if (n>0) {
    double cost;
    for(int i = 0 ; i < n; i++) {
      c=fgets(bufS,1024,fpF);
      if(2 == sscanf(bufS,"%s %lf ", tokS, &cost)) {
	std::string synID(tokS);
	_chemicalSynapseCostsMap[synID]=cost;
      }
      else assert(0);
    }
    rval=true;
  }
  return rval;
}

unsigned long long Params::resetMask(FILE* fpF, std::vector<SegmentDescriptor::SegmentKeyData>& maskVector)
{
  maskVector.clear();
  char bufS[1024];
  char* c=fgets(bufS,1024,fpF);
  std::istringstream is(bufS);
  std::string tokS;
  is>>tokS;
  while(!is.eof()) {
    maskVector.push_back(_segmentDescriptor.getSegmentKeyData(tokS));
    is>>tokS;
  }
  return _segmentDescriptor.getMask(maskVector);
}

