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
#include "StringUtils.h"
#include "NumberUtils.h"

#include <string.h>
#include <sstream>
#include <vector>
#include <iostream>
#include <algorithm>
#include <valarray>
#include <functional>
#include <numeric>
#include <climits>
#include <string>

//x = row, y=col
//WIDTH=#col, HEIGHT=#row
#ifndef Map1Dindex
#define Map1Dindex(x,y, WIDTH) ((y)+(x)*(WIDTH))
#endif

#ifndef Find2Dindex
#define Find2Dindex(x,y, i, WIDTH) \
		do{\
			(y) = (i) % (WIDTH);\
			(x) = (i) / (WIDTH);\
		}while (0)
#endif

#define LENGTH_LINE_MAX 10240
// the maximum length of the name given to each FieldName as part of the key
// helping to identify the 'component' in a branch
#define LENGTH_TOKEN_MAX 256
// the maximum length of the name given to each "Type" in GSL
#define LENGTH_IDNAME_MAX 256

Params::Params()
    : _bondK0(0),
      _bondR0(0),
      _nBondTypes(0),
      _angleK0(0),
      _angleR0(0),
      _nAngleTypes(0),
      _ljEps(0),
      _ljR0(0),
      _nLJTypes(0),
      _radiiMask(0),
      _SIParamsMask(0),
      _compartmentVariableTargetsMask(0),
      _channelTargetsMask(0),
      _electricalSynapseTargetsMask1(0),
      _electricalSynapseTargetsMask2(0),
      _chemicalSynapseTargetsMask1(0),
      _chemicalSynapseTargetsMask2(0),
			_bidirectionalConnectionTargetsMask1(0),
			_bidirectionalConnectionTargetsMask2(0),
      _SIParams(false),
      _compartmentVariables(false),
      _channels(false),
      _electricalSynapses(false),
      _chemicalSynapses(false),
      _bidirectionalConnections(false)
{
}

Params::Params(Params const& p)
    : _bondK0(0),
      _bondR0(0),
      _nBondTypes(p._nBondTypes),
      _angleK0(0),
      _angleR0(0),
      _nAngleTypes(p._nAngleTypes),
      _ljEps(0),
      _ljR0(0),
      _nLJTypes(p._nLJTypes),
      _radiiMask(p._radiiMask),
      _SIParamsMask(p._SIParamsMask),
      _compartmentVariableTargetsMask(p._compartmentVariableTargetsMask),
      _channelTargetsMask(p._channelTargetsMask),
      _electricalSynapseTargetsMask1(p._electricalSynapseTargetsMask1),
      _electricalSynapseTargetsMask2(p._electricalSynapseTargetsMask2),
      _chemicalSynapseTargetsMask1(p._chemicalSynapseTargetsMask1),
      _chemicalSynapseTargetsMask2(p._chemicalSynapseTargetsMask2),
			_bidirectionalConnectionTargetsMask1(p._bidirectionalConnectionTargetsMask1),
			_bidirectionalConnectionTargetsMask2(p._bidirectionalConnectionTargetsMask2),
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
			_bidirectionalConnectionTargetsMap(p._bidirectionalConnectionTargetsMap),
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
			_bidirectionalConnections(p._bidirectionalConnections),
      _segmentDescriptor(p._segmentDescriptor)
{
  _bondK0 = new double[_nBondTypes];
  _bondR0 = new double[_nBondTypes];
  for (int i = 0; i < _nBondTypes; i++)
  {
    _bondK0[i] = p._bondK0[i];
    _bondR0[i] = p._bondR0[i];
  }
  _angleK0 = new double[_nAngleTypes];
  _angleR0 = new double[_nAngleTypes];
  for (int i = 0; i < _nAngleTypes; i++)
  {
    _angleK0[i] = p._angleK0[i];
    _angleR0[i] = p._angleR0[i];
  }
  _ljEps = new double[_nLJTypes];
  _ljR0 = new double[_nLJTypes];
  for (int i = 0; i < _nLJTypes; i++)
  {
    _ljEps[i] = p._ljEps[i];
    _ljR0[i] = p._ljR0[i];
  }
}

Params::Params(Params& p)
    : _bondK0(0),
      _bondR0(0),
      _nBondTypes(p._nBondTypes),
      _angleK0(0),
      _angleR0(0),
      _nAngleTypes(p._nAngleTypes),
      _ljEps(0),
      _ljR0(0),
      _nLJTypes(p._nLJTypes),
      _radiiMask(p._radiiMask),
      _SIParamsMask(p._SIParamsMask),
      _compartmentVariableTargetsMask(p._compartmentVariableTargetsMask),
      _channelTargetsMask(p._channelTargetsMask),
      _electricalSynapseTargetsMask1(p._electricalSynapseTargetsMask1),
      _electricalSynapseTargetsMask2(p._electricalSynapseTargetsMask2),
      _chemicalSynapseTargetsMask1(p._chemicalSynapseTargetsMask1),
      _chemicalSynapseTargetsMask2(p._chemicalSynapseTargetsMask2),
			_bidirectionalConnectionTargetsMask1(p._bidirectionalConnectionTargetsMask1),
			_bidirectionalConnectionTargetsMask2(p._bidirectionalConnectionTargetsMask2),
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
			_bidirectionalConnectionTargetsMap(p._bidirectionalConnectionTargetsMap),
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
			_bidirectionalConnections(p._bidirectionalConnections),
      _segmentDescriptor(p._segmentDescriptor)
{
  _bondK0 = new double[_nBondTypes];
  _bondR0 = new double[_nBondTypes];
  for (int i = 0; i < _nBondTypes; i++)
  {
    _bondK0[i] = p._bondK0[i];
    _bondR0[i] = p._bondR0[i];
  }
  _angleK0 = new double[_nAngleTypes];
  _angleR0 = new double[_nAngleTypes];
  for (int i = 0; i < _nAngleTypes; i++)
  {
    _angleK0[i] = p._angleK0[i];
    _angleR0[i] = p._angleR0[i];
  }
  _ljEps = new double[_nLJTypes];
  _ljR0 = new double[_nLJTypes];
  for (int i = 0; i < _nLJTypes; i++)
  {
    _ljEps[i] = p._ljEps[i];
    _ljR0[i] = p._ljR0[i];
  }
}

Params::~Params()
{
  delete[] _bondR0;
  delete[] _bondK0;
  delete[] _angleR0;
  delete[] _angleK0;
  delete[] _ljR0;
  delete[] _ljEps;
}

// GOAL: read DevParams.par
void Params::readDevParams(const std::string& fname)
{
  FILE* fpF = fopen(fname.c_str(), "r");
	if (fpF == NULL)
	{
		std::cerr << "File " << fname << " not found.\n";
		assert(fpF);
	}
  skipHeader(fpF);
  assert(readBondParams(fpF));
  // assert(readAngleParams(fpF));
  assert(readLJParams(fpF));
  assert(readSIParams(fpF));
  assert(readRadii(fpF));
  fclose(fpF);
}

// GOAL: read DetParams.par
void Params::readDetParams(const std::string& fname)
{
  FILE* fpF = fopen(fname.c_str(), "r");
	if (fpF == NULL)
	{
		std::cerr << "File " << fname << " not found.\n";
		assert(fpF);
	}
  skipHeader(fpF);
  assert(readRadii(fpF));
  assert(readTouchTables(fpF));
  fclose(fpF);
}

void Params::readCptParams(const std::string& fname)
{
  FILE* fpF = fopen(fname.c_str(), "r");
	if (fpF == NULL)
	{
		std::cerr << "File " << fname << " not found.\n";
		assert(fpF);
	}
  skipHeader(fpF);
  assert(readCompartmentVariableTargets(fpF));
  assert(readCompartmentVariableCosts(fpF));
  readModelParams(fpF, "COMPARTMENT_VARIABLE_PARAMS", _compartmentParamsMasks,
                  _compartmentParamsMap, _compartmentArrayParamsMap);
  fclose(fpF);
}

void Params::readChanParams(const std::string& fname)
{
  FILE* fpF = fopen(fname.c_str(), "r");
	if (fpF == NULL)
	{
		std::cerr << "File " << fname << " not found.\n";
		assert(fpF);
	}
  skipHeader(fpF);
  readChannelTargets(fpF);
  readChannelCosts(fpF);
  readModelParams(fpF, "CHANNEL_PARAMS", _channelParamsMasks, _channelParamsMap,
                  _channelArrayParamsMap);
  fclose(fpF);
}

void Params::readSynParams(const std::string& fname)
{
  FILE* fpF = fopen(fname.c_str(), "r");
	if (fpF == NULL)
	{
		std::cerr << "File " << fname << " not found.\n";
		assert(fpF);
	}
  skipHeader(fpF);

  std::string keyword;
  keyword = std::string("ELECTRICAL_SYNAPSE_TARGETS");
  if (isGivenKeywordNext(fpF, keyword))
  {
    assert(readElectricalSynapseTargets(fpF));
    assert(readElectricalSynapseCosts(fpF));
  }
  keyword = std::string("BIDIRECTIONAL_CONNECTION_TARGETS");
  if (isGivenKeywordNext(fpF, keyword))
  {
    assert(readBidirectionalConnectionTargets(fpF));
    assert(readBidirectionalConnectionCosts(fpF));
  }
  keyword = std::string("CHEMICAL_SYNAPSE_TARGETS");
  if (isGivenKeywordNext(fpF, keyword))
  {
    assert(readChemicalSynapseTargets(fpF));
    assert(readChemicalSynapseCosts(fpF));
  }
  // readElectricalSynapseTargets(fpF);
  // readElectricalSynapseCosts(fpF);
  // readBidirectionalConnectionTargets(fpF);
  // readBidirectionalConnectionCosts(fpF);
  // readChemicalSynapseTargets(fpF);
  // readChemicalSynapseCosts(fpF);
  assert(readPreSynapticPointTargets(fpF));
  fclose(fpF);
}

bool Params::symmetricElectricalSynapseTargets(key_size_t key1, key_size_t key2)
{
  return (
      _segmentDescriptor.getSegmentKey(key1, _electricalSynapseTargetsMask1) ==
      _segmentDescriptor.getSegmentKey(key2, _electricalSynapseTargetsMask2));
}

bool Params::symmetricBidirectionalConnectionTargets(key_size_t key1,
                                                     key_size_t key2)
{
  return (_segmentDescriptor.getSegmentKey(
              key1, _bidirectionalConnectionTargetsMask1) ==
          _segmentDescriptor.getSegmentKey(
              key2, _bidirectionalConnectionTargetsMask2));
}

double Params::getRadius(key_size_t key)
{
  double rval = 0.0;
  std::map<key_size_t, double>::iterator iter =
      _radiiMap.find(_segmentDescriptor.getSegmentKey(key, _radiiMask));
  if (iter != _radiiMap.end()) rval = iter->second;
  return rval;
}

SIParameters Params::getSIParams(key_size_t key1, key_size_t key2)
{
  SIParameters rval;
  if (_SIParams)
  {
    bool rvalSet = false;

    std::map<key_size_t, std::map<key_size_t, SIParameters> >::iterator iter1 =
        _SIParamsMap.find(
            _segmentDescriptor.getSegmentKey(key1, _SIParamsMask));
    if (iter1 != _SIParamsMap.end())
    {
      std::map<key_size_t, SIParameters>::iterator iter2 = iter1->second.find(
          _segmentDescriptor.getSegmentKey(key2, _SIParamsMask));
      if (iter2 != iter1->second.end())
      {
        rval = iter2->second;
        rvalSet = true;
      }
    }
  }
  return rval;
}

std::list<std::string> const* Params::getCompartmentVariableTargets(
    key_size_t key)
{
  std::list<std::string>* rval = 0;
  if (_compartmentVariables)
  {
    std::map<key_size_t, std::list<std::string> >::iterator miter =
        _compartmentVariableTargetsMap.find(_segmentDescriptor.getSegmentKey(
            key, _compartmentVariableTargetsMask));
    if (miter != _compartmentVariableTargetsMap.end())
    {
      rval = &miter->second;
    }
  }
  return rval;
}

std::list<Params::ChannelTarget>* Params::getChannelTargets(key_size_t key)
{
  std::list<Params::ChannelTarget>* rval = 0;
  if (_channels)
  {
    std::map<key_size_t, std::list<Params::ChannelTarget> >::iterator miter =
        _channelTargetsMap.find(
            _segmentDescriptor.getSegmentKey(key, _channelTargetsMask));
    if (miter != _channelTargetsMap.end())
    {
      rval = &miter->second;
    }
  }
  return rval;
}

std::list<Params::ElectricalSynapseTarget>*
    Params::getElectricalSynapseTargets(key_size_t key1, key_size_t key2)
{
  std::list<Params::ElectricalSynapseTarget>* rval = 0;
  if (_electricalSynapses)
  {  // if there is information about what branch can connect with what branch
    // to form a bidirectional connection
    // then
    std::map<key_size_t,
             std::map<key_size_t,
                      std::list<Params::ElectricalSynapseTarget> > >::
        iterator miter1 =
            _electricalSynapseTargetsMap.find(_segmentDescriptor.getSegmentKey(
                key1, _electricalSynapseTargetsMask1));
    if (miter1 != _electricalSynapseTargetsMap.end())
    {
      std::map<key_size_t,
               std::list<Params::ElectricalSynapseTarget> >::iterator
          miter2 = miter1->second.find(_segmentDescriptor.getSegmentKey(
              key2, _electricalSynapseTargetsMask2));
      if (miter2 != miter1->second.end())
      {
        rval = &miter2->second;
      }
    }
  }
  return rval;
}

std::list<Params::BidirectionalConnectionTarget>*
    Params::getBidirectionalConnectionTargets(key_size_t key1, key_size_t key2)
{
  std::list<Params::BidirectionalConnectionTarget>* rval = 0;
  if (_bidirectionalConnections)
  {  // if there is information about what branch can connect with what branch
    // to form a bidirectional connection
    // then
    std::map<
        key_size_t,
        std::map<key_size_t,
                 std::list<Params::BidirectionalConnectionTarget> > >::
        iterator miter1 = _bidirectionalConnectionTargetsMap.find(
            _segmentDescriptor.getSegmentKey(
                key1, _bidirectionalConnectionTargetsMask1));
    if (miter1 != _bidirectionalConnectionTargetsMap.end())
    {
      std::map<
          key_size_t,
          std::list<Params::BidirectionalConnectionTarget> >::iterator
          miter2 = miter1->second.find(_segmentDescriptor.getSegmentKey(
              key2, _bidirectionalConnectionTargetsMask2));
      if (miter2 != miter1->second.end())
      {
        rval = &miter2->second;
      }
    }
  }
  return rval;
}

std::list<Params::ChemicalSynapseTarget>* Params::getChemicalSynapseTargets(
    key_size_t key1, key_size_t key2)
{
  std::list<Params::ChemicalSynapseTarget>* rval = 0;
  if (_chemicalSynapses)
  {
    std::map<key_size_t,
             std::map<key_size_t, std::list<Params::ChemicalSynapseTarget> > >::
        iterator miter1 =
            _chemicalSynapseTargetsMap.find(_segmentDescriptor.getSegmentKey(
                key1, _chemicalSynapseTargetsMask1));
    if (miter1 != _chemicalSynapseTargetsMap.end())
    {
      std::map<key_size_t, std::list<Params::ChemicalSynapseTarget> >::iterator
          miter2 = miter1->second.find(_segmentDescriptor.getSegmentKey(
              key2, _chemicalSynapseTargetsMask2));
      if (miter2 != miter1->second.end())
      {
        rval = &miter2->second;
      }
    }
  }
  return rval;
}

std::string Params::getPreSynapticPointTarget(std::string chemicalSynapseType)
{
  std::map<std::string, std::string>::iterator iter =
      _preSynapticPointTargetsMap.find(chemicalSynapseType);
  assert(iter != _preSynapticPointTargetsMap.end());
  return (iter->second);
}

std::list<std::string>& Params::getPreSynapticPointSynapseTypes(
    std::string targetType)
{
  std::map<std::string, std::list<std::string> >::iterator iter =
      _preSynapticPointSynapseMap.find(targetType);
  assert(iter != _preSynapticPointSynapseMap.end());
  return iter->second;
}

bool Params::isCompartmentVariableTarget(key_size_t key, std::string type)
{
  bool rval = false;
  if (_compartmentVariables)
  {
    std::map<key_size_t, std::list<std::string> >::iterator miter =
        _compartmentVariableTargetsMap.find(_segmentDescriptor.getSegmentKey(
            key, _compartmentVariableTargetsMask));
    if (miter != _compartmentVariableTargetsMap.end())
    {
      std::list<std::string>::iterator titer = miter->second.begin(),
                                       tend = miter->second.end();
      for (; titer != tend; ++titer)
      {
        if (*titer == type)
        {
          rval = true;
          break;
        }
      }
    }
  }
  return rval;
}

bool Params::isChannelTarget(key_size_t key)
{
  bool rval = false;
  if (_channels)
  {
    std::map<key_size_t, std::list<Params::ChannelTarget> >::iterator miter =
        _channelTargetsMap.find(
            _segmentDescriptor.getSegmentKey(key, _channelTargetsMask));
    if (miter != _channelTargetsMap.end())
    {
      rval = true;
    }
  }
  return rval;
}

bool Params::isElectricalSynapseTarget(key_size_t key1, key_size_t key2,
                                       bool autapses)
{
  bool rval;
  rval = isGapJunctionTarget(key1, key2, autapses) ||
         isBidirectionalConnectionTarget(key1, key2, autapses);
  return rval;
}
bool Params::isElectricalSynapseTarget(key_size_t key)
{
  bool rval;
  rval = isGapJunctionTarget(key) || isBidirectionalConnectionTarget(key);
  return rval;
}
bool Params::isGapJunctionTarget(key_size_t key1, key_size_t key2,
                                 bool autapses)
{
  bool rval = false;
  if (_electricalSynapses &&
      (key1 < key2 || !symmetricElectricalSynapseTargets(key1, key2)) &&
      (autapses ||
       _segmentDescriptor.getNeuronIndex(key1) !=
           _segmentDescriptor.getNeuronIndex(key2)))
  {
    std::map<key_size_t,
             std::map<key_size_t,
                      std::list<Params::ElectricalSynapseTarget> > >::iterator
        miter =
            _electricalSynapseTargetsMap.find(_segmentDescriptor.getSegmentKey(
                key1, _electricalSynapseTargetsMask1));
    if (miter != _electricalSynapseTargetsMap.end())
    {
      std::map<key_size_t,
               std::list<Params::ElectricalSynapseTarget> >::iterator miter2 =
          miter->second.find(_segmentDescriptor.getSegmentKey(
              key2, _electricalSynapseTargetsMask2));
      if (miter2 != miter->second.end())
      {
        rval = true;
      }
    }
  }
  return rval;
}

bool Params::isGapJunctionTarget(key_size_t key)
{
  bool rval = false;
  if (_electricalSynapses)
  {
    std::map<
        key_size_t,
        std::map<key_size_t, std::list<Params::ElectricalSynapseTarget> > >::
        const_iterator miter = _electricalSynapseTargetsMap.begin(),
                       mend = _electricalSynapseTargetsMap.end();
    rval = (_electricalSynapseTargetsMap.find(_segmentDescriptor.getSegmentKey(
                key, _electricalSynapseTargetsMask1)) != mend ||
            _electricalSynapseTargetsMap.find(_segmentDescriptor.getSegmentKey(
                key, _electricalSynapseTargetsMask2)) != mend);
    for (; miter != mend && !rval; ++miter)
    {
      std::map<key_size_t,
               std::list<Params::ElectricalSynapseTarget> >::const_iterator
          mend2 = miter->second.end();
      rval = (miter->second.find(_segmentDescriptor.getSegmentKey(
                  key, _electricalSynapseTargetsMask1)) != mend2 ||
              miter->second.find(_segmentDescriptor.getSegmentKey(
                  key, _electricalSynapseTargetsMask2)) != mend2);
    }
  }
  return rval;
}

bool Params::isBidirectionalConnectionTarget(key_size_t key1, key_size_t key2,
                                             bool autapses)
{
  bool rval = false;
  // bi-direction
  // for (int direction = 0; direction <= 1 && !rval; ++direction)
  //{
  if (_bidirectionalConnections &&
      (key1 < key2 || !symmetricBidirectionalConnectionTargets(key1, key2)) &&
      (autapses ||
       _segmentDescriptor.getNeuronIndex(key1) !=
           _segmentDescriptor.getNeuronIndex(key2)))
  {
    std::map<
        key_size_t,
        std::map<key_size_t,
                 std::list<Params::BidirectionalConnectionTarget> > >::iterator
        miter = _bidirectionalConnectionTargetsMap.find(
            _segmentDescriptor.getSegmentKey(
                key1, _bidirectionalConnectionTargetsMask1));
    if (miter != _bidirectionalConnectionTargetsMap.end())
    {
      std::map<key_size_t,
               std::list<Params::BidirectionalConnectionTarget> >::iterator
          miter2 = miter->second.find(_segmentDescriptor.getSegmentKey(
              key2, _bidirectionalConnectionTargetsMask2));
      if (miter2 != miter->second.end())
      {
        rval = true;
      }
    }
  }
  /* key_size_t tmp = key1;
   key1 = key2;
   key2 = tmp;
 }*/
  return rval;
}

bool Params::isBidirectionalConnectionTarget(key_size_t key)
{
  bool rval = false;
  if (_bidirectionalConnections)
  {
    std::map<key_size_t,
             std::map<key_size_t,
                      std::list<Params::BidirectionalConnectionTarget> > >::
        const_iterator miter = _bidirectionalConnectionTargetsMap.begin(),
                       mend = _bidirectionalConnectionTargetsMap.end();
    rval = (_bidirectionalConnectionTargetsMap.find(
                _segmentDescriptor.getSegmentKey(
                    key, _bidirectionalConnectionTargetsMask1)) != mend ||
            _bidirectionalConnectionTargetsMap.find(
                _segmentDescriptor.getSegmentKey(
                    key, _bidirectionalConnectionTargetsMask2)) != mend);
    for (; miter != mend && !rval; ++miter)
    {
      std::map<key_size_t, std::list<Params::BidirectionalConnectionTarget> >::
          const_iterator mend2 = miter->second.end();
      rval = (miter->second.find(_segmentDescriptor.getSegmentKey(
                  key, _bidirectionalConnectionTargetsMask1)) != mend2 ||
              miter->second.find(_segmentDescriptor.getSegmentKey(
                  key, _bidirectionalConnectionTargetsMask2)) != mend2);
    }
  }
  return rval;
}

bool Params::isChemicalSynapseTarget(key_size_t key1, key_size_t key2,
                                     bool autapses)
{
  bool rval = false;
  // uni-direction
  if (_chemicalSynapses && (autapses ||
                            _segmentDescriptor.getNeuronIndex(key1) !=
                                _segmentDescriptor.getNeuronIndex(key2)))
  {
    std::map<key_size_t,
             std::map<key_size_t, std::list<Params::ChemicalSynapseTarget> > >::
        iterator miter =
            _chemicalSynapseTargetsMap.find(_segmentDescriptor.getSegmentKey(
                key1, _chemicalSynapseTargetsMask1));
    if (miter != _chemicalSynapseTargetsMap.end())
    {
      std::map<key_size_t, std::list<Params::ChemicalSynapseTarget> >::iterator
          miter2 = miter->second.find(_segmentDescriptor.getSegmentKey(
              key2, _chemicalSynapseTargetsMask2));
      if (miter2 != miter->second.end())
      {
        rval = true;
      }
    }
  }
  return rval;
}

bool Params::isChemicalSynapseTarget(key_size_t key)
{
  bool rval = false;
  if (_chemicalSynapses)
  {
    std::map<key_size_t,
             std::map<key_size_t, std::list<Params::ChemicalSynapseTarget> > >::
        const_iterator miter = _chemicalSynapseTargetsMap.begin(),
                       mend = _chemicalSynapseTargetsMap.end();
    rval = (_chemicalSynapseTargetsMap.find(_segmentDescriptor.getSegmentKey(
                key, _chemicalSynapseTargetsMask1)) != mend ||
            _chemicalSynapseTargetsMap.find(_segmentDescriptor.getSegmentKey(
                key, _chemicalSynapseTargetsMask2)) != mend);
    for (; miter != mend && !rval; ++miter)
    {
      std::map<key_size_t,
               std::list<Params::ChemicalSynapseTarget> >::const_iterator
          mend2 = miter->second.end();
      rval = (miter->second.find(_segmentDescriptor.getSegmentKey(
                  key, _chemicalSynapseTargetsMask1)) != mend2 ||
              miter->second.find(_segmentDescriptor.getSegmentKey(
                  key, _chemicalSynapseTargetsMask2)) != mend2);
    }
  }
  return rval;
}

double Params::getCompartmentVariableCost(std::string compartmentVariableId)
{
  double rval = 0.0;
  std::map<std::string, double>::iterator miter =
      _compartmentVariableCostsMap.find(compartmentVariableId);
  if (miter != _compartmentVariableCostsMap.end())
  {
    rval = miter->second;
  }
  else
  {
    std::cerr << "Params : Unspecified CompartmentVariable Cost! ID : "
              << compartmentVariableId << std::endl;
    exit(EXIT_FAILURE);
  }
  return rval;
}

double Params::getChannelCost(std::string channelId)
{
  double rval = 0.0;
  std::map<std::string, double>::iterator miter =
      _channelCostsMap.find(channelId);
  if (miter != _channelCostsMap.end())
  {
    rval = miter->second;
  }
  else
  {
    std::cerr << "Params : Unspecified Channel Cost! ID : " << channelId
              << std::endl;
    exit(EXIT_FAILURE);
  }
  return rval;
}

double Params::getElectricalSynapseCost(std::string electricalSynapseId)
{
  double rval = 0.0;
  std::map<std::string, double>::iterator miter =
      _electricalSynapseCostsMap.find(electricalSynapseId);
  if (miter != _electricalSynapseCostsMap.end())
  {
    rval = miter->second;
  }
  else
  {
    std::cerr << "Params : Unspecified Electrical Synapse Cost! ID : "
              << electricalSynapseId << std::endl;
    exit(EXIT_FAILURE);
  }
  return rval;
}

double Params::getBidirectionalConnectionCost(
    std::string bidirectionalConnectionId)
{
  double rval = 0.0;
  std::map<std::string, double>::iterator miter =
      _bidirectionalConnectionCostsMap.find(bidirectionalConnectionId);
  if (miter != _bidirectionalConnectionCostsMap.end())
  {
    rval = miter->second;
  }
  else
  {
    std::cerr << "Params : Unspecified Bidirectional Connection Cost! ID : "
              << bidirectionalConnectionId << std::endl;
    exit(EXIT_FAILURE);
  }
  return rval;
}

double Params::getChemicalSynapseCost(std::string chemicalSynapseId)
{
  double rval = 0.0;
  std::map<std::string, double>::iterator miter =
      _chemicalSynapseCostsMap.find(chemicalSynapseId);
  if (miter != _chemicalSynapseCostsMap.end())
  {
    rval = miter->second;
  }
  else
  {
    std::cerr << "Params : Unspecified Chemical Synapse Cost! ID: "
              << chemicalSynapseId << std::endl;
    exit(EXIT_FAILURE);
  }
  return rval;
}

// INPUT:
//    modelType = COMPARTMENT | SYNAPSE | CHANNEL
//    nodeType  = the name for the nodetype passed via nodekine in Layer
//    statement in GSL
//    key       = the key associated with the branch's compartment from that we
//    want to retrieve
// OUTPUT:
//    modelParams = the list of pair (parameter, value) for the give nodeType
//        such 'value' are input via the *params.par file
void Params::getModelParams(
    ModelType modelType, std::string nodeType, key_size_t key,
    std::list<std::pair<std::string, dyn_var_t> >& modelParams)
{
  std::map<std::string, unsigned long long>* modelParamsMasks;
  std::map<
      std::string,
      std::map<key_size_t, std::list<std::pair<std::string, dyn_var_t> > > >*
      modelParamsMap;

  switch (modelType)
  {
    case COMPARTMENT:
      modelParamsMasks = &_compartmentParamsMasks;
      modelParamsMap = &_compartmentParamsMap;
      break;
    case CHANNEL:
      modelParamsMasks = &_channelParamsMasks;
      modelParamsMap = &_channelParamsMap;
      break;
  }

  modelParams.clear();
  std::map<std::string,
           std::map<key_size_t,
                    std::list<std::pair<std::string, dyn_var_t> > > >::iterator
      iter1 = modelParamsMap->find(nodeType);
  if (iter1 != modelParamsMap->end())
  {
    std::map<std::string, unsigned long long>::iterator miter =
        modelParamsMasks->find(nodeType);
    assert(miter != modelParamsMasks->end());
    std::map<key_size_t,
             std::list<std::pair<std::string, dyn_var_t> > >::iterator iter2 =
        (iter1->second)
            .find(_segmentDescriptor.getSegmentKey(key, miter->second));
    if (iter2 != (iter1->second).end())
    {
      modelParams = iter2->second;
    }
  }
}

void Params::getModelArrayParams(
    ModelType modelType, std::string nodeType, key_size_t key,
    std::list<std::pair<std::string, std::vector<dyn_var_t> > >&
        modelArrayParams)
{
  std::map<std::string, unsigned long long>* modelParamsMasks;
  std::map<
      std::string,
      std::map<key_size_t,
               std::list<std::pair<std::string, std::vector<dyn_var_t> > > > >*
      modelArrayParamsMap;

  switch (modelType)
  {
    case COMPARTMENT:
      modelParamsMasks = &_compartmentParamsMasks;
      modelArrayParamsMap = &_compartmentArrayParamsMap;
      break;
    case CHANNEL:
      modelParamsMasks = &_channelParamsMasks;
      modelArrayParamsMap = &_channelArrayParamsMap;
      break;
  }

  modelArrayParams.clear();
  std::map<
      std::string,
      std::map<key_size_t,
               std::list<std::pair<std::string, std::vector<dyn_var_t> > > > >::
      iterator iter1 = modelArrayParamsMap->find(nodeType);
  if (iter1 != modelArrayParamsMap->end())
  {
    std::map<std::string, unsigned long long>::iterator miter =
        modelParamsMasks->find(nodeType);
    assert(miter != modelParamsMasks->end());
    std::map<
        key_size_t,
        std::list<std::pair<std::string, std::vector<dyn_var_t> > > >::iterator
        iter2 = (iter1->second)
                    .find(_segmentDescriptor.getSegmentKey(key, miter->second));
    if (iter2 != (iter1->second).end())
    {
      modelArrayParams = iter2->second;
    }
  }
}

//GOAL: check if the content of a string is a comment-line
// A comment line is a blank line
//  or               a line whose first non-space character is #
bool Params::isCommentLine(std::string& line)
{
  bool rval = false;
  char space = ' ';
	std::string::size_type  index = line.find_first_not_of(space);
  if (index != std::string::npos)
  {
    if (line[index] == '#' or line[index] == '\n') rval = true;
  }
  else if (line.length() == 0)
    rval = true;
  return rval;
}

// Skip the comment lines
//   - including blank line
void Params::jumpOverCommentLine(FILE* fpF)
{
  fpos_t fpos;
  fgetpos(fpF, &fpos);
  char bufS[LENGTH_LINE_MAX];
	char* c = fgets(bufS, LENGTH_LINE_MAX, fpF);
	std::string line(bufS);
	while ((bufS[0] == '\n' or isCommentLine(line)) and !feof(fpF))
	{
		fgetpos(fpF, &fpos);
		c = fgets(bufS, LENGTH_LINE_MAX, fpF);
		line = std::string(bufS);
	}
	fsetpos(fpF, &fpos);
}
// GOAL: return true if the next section is the one
//       matching the given keyword
// ASSUMPTION:
//   1. it expects the next non-comment line is a beginning of a section
//   2. this beginning line has the format
//      KEYWORD NUM
bool Params::isGivenKeywordNext(FILE* fpF, std::string& keyword)
{
  bool rval = false;
  fpos_t fpos;
  fgetpos(fpF, &fpos);
  char bufS[LENGTH_LINE_MAX], tokS[LENGTH_TOKEN_MAX];
  int n;
  jumpOverCommentLine(fpF);
  char* c = fgets(bufS, LENGTH_LINE_MAX, fpF);
  std::string line(bufS);
  if (2 == sscanf(bufS, "%s %d ", tokS, &n))
  {
    std::string btype(tokS);
    if (btype == keyword)
    {
      rval = true;
    }
    else
    {
      std::cerr << "Expected: " << keyword << ", but Found: " << btype
                << std::endl;
    }
  }
  else
  {
    std::cerr << "Syntax of SynParam invalid: expect \n SOME_KEYWORD num-column"
              << std::endl;
    std::cerr << "Read line: " << line << std::endl;
  }
  fsetpos(fpF, &fpos);
  return rval;
}

// GOAL: read the keyword in the next section
// ASSUMPTION:
//   1. it expects the next non-comment line is a beginning of a section
//   2. this beginning line has the format
//      KEYWORD NUM
// which can be used to identify the section
std::string Params::findNextKeyword(FILE* fpF)
{
  std::string rval;
  fpos_t fpos;
  fgetpos(fpF, &fpos);
  char bufS[LENGTH_LINE_MAX], tokS[LENGTH_TOKEN_MAX];
  int n;
  char* c = fgets(bufS, LENGTH_LINE_MAX, fpF);
  std::string line(bufS);
  if (2 == sscanf(bufS, "%s %d ", tokS, &n))
  {
    rval = std::string(tokS);
  }
  else
  {
    std::cerr << "Syntax of SynParam invalid: expect \n SOME_KEYWORD num-column"
              << std::endl;
  }
  fsetpos(fpF, &fpos);
  return rval;
}

// GOAL: read section    NBONDTYPES
//  Each line maps to the bonding for one branchtype (1=soma,2=axon...)
//  Make sure it equals to the total possible values for branchtype
bool Params::readBondParams(FILE* fpF)
{
  bool rval = true;
  int n = _nBondTypes = 0;
  char bufS[LENGTH_LINE_MAX], tokS[LENGTH_TOKEN_MAX];
  jumpOverCommentLine(fpF);
  char* c = fgets(bufS, LENGTH_LINE_MAX, fpF);
  if (2 == sscanf(bufS, "%s %d ", tokS, &n))
  {
    std::string btype = std::string(tokS);
    std::string expected_btype("NBONDTYPES");
    if (btype == expected_btype)
    {
      _nBondTypes = n;
    }
    else
      rval = false;
  }
  else
    rval = false;

  delete[] _bondR0;
  delete[] _bondK0;
  _bondR0 = _bondK0 = 0;
  if (_nBondTypes > 0)
  {
    _bondR0 = new double[_nBondTypes];
    _bondK0 = new double[_nBondTypes];
    for (int i = 0; i < _nBondTypes; i++)
    {
      jumpOverCommentLine(fpF);
      char* c = fgets(bufS, LENGTH_LINE_MAX, fpF);
      if (2 != sscanf(bufS, "%lf %lf ", &_bondK0[i], &_bondR0[i]))
      {
        rval = false;
        assert(0);
      }
    }
  }
  else
    rval = false;
  return rval;
}

bool Params::readAngleParams(FILE* fpF)
{
  bool rval = true;
  int n = _nAngleTypes = 0;
  char bufS[LENGTH_LINE_MAX], tokS[LENGTH_TOKEN_MAX];
  jumpOverCommentLine(fpF);
  char* c = fgets(bufS, LENGTH_LINE_MAX, fpF);
  if (2 == sscanf(bufS, "%s %d ", tokS, &n))
  {
    std::string btype(tokS);
    std::string expected_btype("NANGLETYPES");
    if (btype == expected_btype)
    {
      //            if (!strcmp("NANGLETYPES", tokS)) {
      _nAngleTypes = n;
    }
    else
      rval = false;
  }
  else
    rval = false;
  delete[] _angleR0;
  delete[] _angleK0;
  _angleR0 = _angleK0 = 0;
  if (_nAngleTypes > 0)
  {
    _angleR0 = new double[_nAngleTypes];
    _angleK0 = new double[_nAngleTypes];
    for (int i = 0; i < _nAngleTypes; i++)  // for each line
    {
      jumpOverCommentLine(fpF);
      c = fgets(bufS, LENGTH_LINE_MAX, fpF);
      if (2 != sscanf(bufS, "%lf %lf ", &_angleK0[i], &_angleR0[i]))
      {
        rval = false;
        assert(0);
      }
      _angleR0[i] *= (M_PI / 180.0);
    }
  }
  else
    rval = false;
  return rval;
}

// GOAL: read section    NREPULSETYPES
bool Params::readLJParams(FILE* fpF)
{
  /*
NREPULSETYPES 6
0.0 1.0
0.0 1.0
0.0 1.0
0.0 1.0
0.0 1.0
0.0 1.0
   */
  bool rval = true;
  int n = _nLJTypes = 0;
  char bufS[LENGTH_LINE_MAX], tokS[LENGTH_TOKEN_MAX];
  jumpOverCommentLine(fpF);
  char* c = fgets(bufS, LENGTH_LINE_MAX, fpF);
  if (2 == sscanf(bufS, "%s %d ", tokS, &n))
  {
    std::string btype(tokS);
    std::string expected_btype1("NLJTYPES");
    std::string expected_btype2("NREPULSETYPES");
    if (btype == expected_btype1 || btype == expected_btype2)
    {
      _nLJTypes = n;
    }
    else
      rval = false;
  }
  else
    rval = false;

  delete[] _ljR0;
  delete[] _ljEps;
  _ljR0 = _ljEps = 0;
  if (_nLJTypes > 0)
  {
    _ljR0 = new double[_nLJTypes];
    _ljEps = new double[_nLJTypes];
    for (int i = 0; i < _nLJTypes; i++)  // for each line
    {
      jumpOverCommentLine(fpF);
      c = fgets(bufS, LENGTH_LINE_MAX, fpF);
      if (2 != sscanf(bufS, "%lf %lf ", &_ljEps[i], &_ljR0[i]))
      {
        rval = false;
        assert(0);
      }
      _ljEps[i] = sqrt(_ljEps[i]);
    }
  }
  else
    rval = false;
  return rval;
}

// GOAL: read section    RADII
bool Params::readRadii(FILE* fpF)
{
  /* Example:
   RADII 2
   BRANCHTYPE
   0 0.001
   1 0.002
   */
  bool rval = true;
  _radiiMask = 0;
  _radiiMap.clear();
  int n = 0;
  char bufS[LENGTH_LINE_MAX], tokS[LENGTH_TOKEN_MAX];
  jumpOverCommentLine(fpF);
  char* c = fgets(bufS, LENGTH_LINE_MAX, fpF);  // read line: RADII 2
  if (2 == sscanf(bufS, "%s %d ", tokS, &n))
  {
    std::string btype(tokS);
    std::string expected_btype("RADII");
    if (btype == expected_btype)
    {
      // do nothing
    }
    else
      rval = false;
  }
  else
    rval = false;

  if (n > 0)
  {
    std::vector<SegmentDescriptor::SegmentKeyData> maskVector;
    _radiiMask = resetMask(fpF, maskVector);  // generate maskVector based on
    // the information
    // on the next line: BRANCHTYPE
    double radius;
    unsigned int sz = maskVector.size();  // number of given fieldnames
    assert(sz);
    unsigned int* ids = new unsigned int[sz];

    for (int i = 0; i < n; i++)  // for each line
    {
      jumpOverCommentLine(fpF);
      for (unsigned int j = 0; j < sz;
           ++j)  // read the values of the associated fieldnames
      {
        if (1 != fscanf(fpF, "%d", &ids[j]))
        {
          rval = false;
          assert(0);
        }
        if (maskVector[j] == SegmentDescriptor::branchType)
          ids[j] = ids[j] - 1;  // make sure the BRANCHTYPE is 0-based
      }  // these values help to identify which branch in which neuron to get
      // the paramater mapping

      c = fgets(bufS, LENGTH_LINE_MAX, fpF);  // read the parameter
      //  to be mapped to the given branch
      //  NOTE: The mapping happens only at the branch-level, not
      //            compartment-level inside the branch
      //        To get to compartment-level, we need to know the index
      //        which can be changed depend on how we specify
      //        #-of-compartment per branch, so this is not easy to deal with
      //        unless we fix the #-of-comparment per branch
      if (1 != sscanf(bufS, "%lf", &radius))
      {
        rval = false;
        assert(0);
      }
      _radiiMap[_segmentDescriptor.getSegmentKey(maskVector, &ids[0])] = radius;
    }
    delete[] ids;
  }
  else
    rval = false;
  return rval;
}

// GOAL: read section		TOUCH_TABLES 
//  which tell what information would be used for touch-detection between any 2 capsules
//  Typically, it is based on 2 informations: NEURON_INDEX and BRANCHTYPE
bool Params::readTouchTables(FILE* fpF)
{
  bool rval = true;
  _touchTableMasks.clear();
  int n = 0;
  char bufS[LENGTH_LINE_MAX], tokS[LENGTH_TOKEN_MAX];
  jumpOverCommentLine(fpF);
  char* c = fgets(bufS, LENGTH_LINE_MAX, fpF);
  if (2 == sscanf(bufS, "%s %d ", tokS, &n))
  {
    std::string btype(tokS);
    std::string expected_btype("TOUCH_TABLES");
    if (btype == expected_btype)
    {
      // do nothing
    }
    else
      rval = false;
  }
  else
    rval = false;
  if (n > 0)
  {
    for (int i = 0; i < n; i++)  // for each line (not counting comment-line)
    {
      jumpOverCommentLine(fpF);
      std::vector<SegmentDescriptor::SegmentKeyData> maskVector;
      resetMask(fpF, maskVector);
      _touchTableMasks.push_back(maskVector);
    }
  }
  else
    rval = false;
  return rval;
}

// GOAL: read section    NSITYPES
//  Each line maps to the SI for one branchtype (1=soma,2=axon...)
//  Make sure it equals to the total possible values for branchtype
bool Params::readSIParams(FILE* fpF)
{
  bool rval = true;
  _SIParams = false;
  _SIParamsMask = 0;
  _SIParamsMap.clear();
  int n = 0;
  char bufS[LENGTH_LINE_MAX], tokS[LENGTH_TOKEN_MAX];
  jumpOverCommentLine(fpF);
  char* c = fgets(bufS, LENGTH_LINE_MAX, fpF);
  if (2 == sscanf(bufS, "%s %d ", tokS, &n))
  {
    std::string btype(tokS);
    std::string expected_btype("NSITYPES");
    if (btype == expected_btype)
    {
      // do nothing
    }
    else
      rval = false;
  }
  else
    rval = false;
  if (n > 0)
  {
    std::vector<SegmentDescriptor::SegmentKeyData> maskVector;
    _SIParamsMask = resetMask(fpF, maskVector);

    double Epsilon, Sigma;
          // for NSITYPES, BRANCHTYPE is used twice
    unsigned int sz = maskVector.size() * 2;
    unsigned int* ids = new unsigned int[sz];

    for (int i = 0; i < n; i++)  // for each non-comment line
    {//each line: BRANCHTYPE BRANCHTYPE Epsilon Sigma
      jumpOverCommentLine(fpF);
      for (unsigned int j = 0; j < sz; ++j)
      {
        if (1 != fscanf(fpF, "%d", &ids[j])) assert(0);
        if (maskVector[j] ==
            SegmentDescriptor::branchType)  // special treatment
          ids[j] = ids[j] - 1;  // make sure the BRANCHTYPE is 0-based
      }
      c = fgets(bufS, LENGTH_LINE_MAX, fpF);
      if (2 != sscanf(bufS, "%lf %lf ", &Epsilon, &Sigma)) assert(0);
      key_size_t key1 = _segmentDescriptor.getSegmentKey(maskVector, &ids[0]);
      key_size_t key2 =
          _segmentDescriptor.getSegmentKey(maskVector, &ids[sz / 2]);
      std::map<key_size_t, std::map<key_size_t, SIParameters> >::iterator iter =
          _SIParamsMap.find(key1);
      if (iter == _SIParamsMap.end())
      {
        std::map<key_size_t, SIParameters> newMap;
        (newMap[key2]).Epsilon = Epsilon;
        (newMap[key2]).Sigma = Sigma;
        _SIParamsMap[key1] = newMap;
      }
      else
      {
        ((*iter).second)[key2].Epsilon = Epsilon;
        ((*iter).second)[key2].Sigma = Sigma;
      }
    }
    delete[] ids;
  }
  else
    rval = false;
  _SIParams = rval;
  return _SIParams;
}

//GOAL: read section COMPARTMENT_VARIABLE_TARGETS
bool Params::readCompartmentVariableTargets(FILE* fpF)
{
  bool rval = true;
  _compartmentVariables = false;
  _compartmentVariableTargetsMask = 0;
  _compartmentVariableTargetsMap.clear();
  int n = 0;
  char bufS[LENGTH_LINE_MAX], tokS[LENGTH_TOKEN_MAX];
  jumpOverCommentLine(fpF);
  char* c = fgets(bufS, LENGTH_LINE_MAX, fpF);
  if (2 == sscanf(bufS, "%s %d ", tokS, &n))
  {
    std::string btype(tokS);
    std::string expected_btype("COMPARTMENT_VARIABLE_TARGETS");
    if (btype == expected_btype)
    {
      // do nothing
    }
    else
      rval = false;
  }
  else
    rval = false;

  if (n > 0)
  {
    std::vector<SegmentDescriptor::SegmentKeyData> maskVector;
    _compartmentVariableTargetsMask = resetMask(fpF, maskVector);
    unsigned int sz = maskVector.size();
    assert(sz);
		for (unsigned int j = 0; j < sz; ++j)
		{
			if (maskVector[j] == SegmentDescriptor::segmentIndex)
			{
				std::cerr << "Params : Targeting compartmentVariables to individual "
					"compartments not supported!" << std::endl;
				return false;
				//exit(EXIT_FAILURE);
			}
		}


    for (int i = 0; i < n; i++)  // for each line, not counting comment-line
    {
      jumpOverCommentLine(fpF);
      std::vector<unsigned int*> v_ids;
      if (checkForSpecialCases(fpF, sz))
			{
        // check if a special case is used
        // A special case can be either:
        //      range [2, 5]
        //      range [2:5]
        //      range [2,5:7]
        // then, create a vector of v_ids
        // NOTE: We planed to support
        //      asterisk *
        //  but then it requires implicit knowledge of the range of value
        //  so we don't support it now

        //unsigned int* ids = new unsigned int[sz];
        std::vector<std::vector<int> > vect_values;
        int total_vect = 1;
        for (unsigned int j = 0; j < sz; ++j)
        {
          std::vector<int> values;
          int val = 0;
          // LIMIT: current only support  special case for BRANCHTYPE
          if (maskVector[j] == SegmentDescriptor::branchType)
          {
            getListofValues(fpF, values);  // assume the next data to read is in
                                           // the form  [...] and it occurs for
                                           // BRANCHTYPE
            /*  if (isAsterisk(fpF))
            {
              assert(0);
            }
            else if (isBracket(fpF, lval, hval))
            {
            }
                                                */
            // make sure the BRANCHTYPE is 0-based
            for (std::vector<int>::iterator it = values.begin();
                 it != values.end(); ++it)
            {
              *it -= 1;
            }
            // std::for_each(values.begin(), values.end(), [](int & d) { d-=1;
            // });
          }
          else
          {
            if (1 != fscanf(fpF, "%d", &val)) assert(0);
            values.push_back(val);
          }
          /*
                   if (1 != fscanf(fpF, "%d", &ids[j])) assert(0);
                   if (maskVector[j] == SegmentDescriptor::branchType)
                   ids[j] = ids[j] - 1;  // make sure the BRANCHTYPE is 0-based
    */
          vect_values.push_back(values);
          total_vect *= values.size();
        }
        {// generate all array elements in the vector
          //unsigned int** pids = new unsigned int* [total_vect];
          for (int jj = 0; jj < total_vect; ++jj)
          {
            //pids[jj] = new unsigned int[sz]();
            //v_ids.push_back(pids[jj]);
            unsigned int *ids = new unsigned int[sz]();
            v_ids.push_back(ids);
          }

          // fill the data
          for (unsigned int jj = 0; jj < sz; jj++)
          {
            int num2clone = 1;
            for (unsigned int xx = jj + 1; xx < sz; xx++)
              num2clone *= vect_values[xx].size();
            int gap = num2clone * vect_values[jj].size();

            for (unsigned int kk = 0; kk < vect_values[jj].size(); kk++)
            {
              for (int xx = (num2clone) * (kk); xx < total_vect; xx += gap)
              {
                std::vector<unsigned int*>::iterator iter,
                    iterstart = v_ids.begin() + xx,
                    iterend = v_ids.begin() + xx + num2clone - 1;
                for (iter = iterstart; iter <= iterend; iter++)
                  (*iter)[jj] = vect_values[jj][kk];
              }
            }
          }
        }
			}
			else
			{ 
				unsigned int* ids = new unsigned int[sz]();
				for (unsigned int j = 0; j < sz; ++j)
				{
					if (1 != fscanf(fpF, "%d", &ids[j])) assert(0);
					if (maskVector[j] == SegmentDescriptor::branchType)
						ids[j] = ids[j] - 1;  // make sure the BRANCHTYPE is 0-based
				}
        // put into v_ids
        v_ids.push_back(ids);
			}

			assert(!feof(fpF));
      std::string myBuf("");
      readMultiLine(myBuf, fpF);
      //std::istringstream is(myBuf);
      /*c = fgets(bufS, LENGTH_LINE_MAX, fpF);
      std::istringstream is(bufS);*/
			//buildCompartmentVariableTargetsMap(maskVector, v_ids, is);
			buildCompartmentVariableTargetsMap(maskVector, v_ids, myBuf);
      // memory clean v_ids
      for (std::vector<unsigned int*>::const_iterator it = v_ids.begin();
           it != v_ids.end(); it++)
      {
        delete *it;
      }
      v_ids.clear();
    }
    //delete[] ids;
  }
  else
    rval = false;
  _compartmentVariables = rval;
  return _compartmentVariables;
}

bool Params::readChannelTargets(FILE* fpF)
{
  /* Example:
  CHANNEL_TARGETS 8
  BRANCHTYPE MTYPE
  1 0 HCN [Voltage] [Voltage] Nat [Voltage] [Voltage]
  */
  _channels = false;
  bool rval = true;
  _channelTargetsMask = 0;
  _channelTargetsMap.clear();
  int n = 0;
  char bufS[LENGTH_LINE_MAX], tokS[LENGTH_TOKEN_MAX];

  jumpOverCommentLine(fpF);
  char* c = fgets(bufS, LENGTH_LINE_MAX, fpF);
  if (2 == sscanf(bufS, "%s %d ", tokS, &n))
  {  // read 'n'
    std::string btype(tokS);
    std::string expected_btype("CHANNEL_TARGETS");
    if (btype == expected_btype)
    {
      // do nothing
    }
    else
      rval = false;
  }
  else
    rval = false;

  if (n > 0)
  {
    std::vector<SegmentDescriptor::SegmentKeyData> maskVector;
    _channelTargetsMask = resetMask(fpF, maskVector);
    unsigned int sz = maskVector.size();
    assert(sz);
    for (unsigned int j = 0; j < sz; j++)
    {  // validate
      if (maskVector[j] == SegmentDescriptor::segmentIndex)
      {
        std::cerr << "Params : Targeting channels to individual compartments "
                     "not supported!" << std::endl;
        return false;
        // exit(EXIT_FAILURE);
      }
    }

    for (int i = 0; i < n; i++)  // for each line, not counting comment-line
    {
      jumpOverCommentLine(fpF);
      std::vector<unsigned int*> v_ids;
      if (checkForSpecialCases(fpF, sz))
      {
        // check if a special case is used
        // A special case can be either:
        //      range [2, 5]
        //      range [2:5]
        //      range [2,5:7]
        // then, create a vector of v_ids
        // NOTE: We planed to support
        //      asterisk *
        //  but then it requires implicit knowledge of the range of value
        //  so we don't support it now

        //unsigned int* ids = new unsigned int[sz];
        std::vector<std::vector<int> > vect_values;
        int total_vect = 1;
        for (unsigned int j = 0; j < sz; ++j)
        {
          std::vector<int> values;
          int val = 0;
          // LIMIT: current only support  special case for BRANCHTYPE
          if (maskVector[j] == SegmentDescriptor::branchType)
          {
            getListofValues(fpF, values);  // assume the next data to read is in
                                           // the form  [...] and it occurs for
                                           // BRANCHTYPE
            /*  if (isAsterisk(fpF))
            {
              assert(0);
            }
            else if (isBracket(fpF, lval, hval))
            {
            }
                                                */
            // make sure the BRANCHTYPE is 0-based
            for (std::vector<int>::iterator it = values.begin();
                 it != values.end(); ++it)
            {
              *it -= 1;
            }
            // std::for_each(values.begin(), values.end(), [](int & d) { d-=1;
            // });
          }
          else
          {
						//int dummy = fscanf(fpF, "%d", &val);
            //char ch[1000];
            //fscanf(fpF," %s", ch);
						//val = atoi(ch);
            if (1 != fscanf(fpF, "%d", &val)) assert(0);
            values.push_back(val);
          }
          /*
                   if (1 != fscanf(fpF, "%d", &ids[j])) assert(0);
                   if (maskVector[j] == SegmentDescriptor::branchType)
                   ids[j] = ids[j] - 1;  // make sure the BRANCHTYPE is 0-based
    */
          vect_values.push_back(values);
          total_vect *= values.size();
        }
        // generate all array elements in the vector
        {
          unsigned int** pids = new unsigned int* [total_vect];
          for (int jj = 0; jj < total_vect; ++jj)
          {
            pids[jj] = new unsigned int[sz]();
            v_ids.push_back(pids[jj]);
          }

          // fill the data
          for (unsigned int jj = 0; jj < sz; jj++)
          {
            int num2clone = 1;
            for (unsigned int xx = jj + 1; xx < sz; xx++)
              num2clone *= vect_values[xx].size();
            int gap = num2clone * vect_values[jj].size();

            for (unsigned int kk = 0; kk < vect_values[jj].size(); kk++)
            {
              for (int xx = (num2clone) * (kk); xx < total_vect; xx += gap)
              {
                std::vector<unsigned int*>::iterator iter,
                    iterstart = v_ids.begin() + xx,
                    iterend = v_ids.begin() + xx + num2clone - 1;
                for (iter = iterstart; iter <= iterend; iter++)
                  (*iter)[jj] = vect_values[jj][kk];
              }
            }
          }
        }
      }
      else
      {
        // v_ids[0] = new unsigned int [sz];
        // unsigned int* ids = v_ids[0];
        unsigned int* ids = new unsigned int[sz]();
        for (unsigned int j = 0; j < sz; ++j)
        {
          if (1 != fscanf(fpF, "%d", &ids[j])) assert(0);
          if (maskVector[j] == SegmentDescriptor::branchType)
          {
            ids[j] = ids[j] - 1;  // make sure the BRANCHTYPE is 0-based
          }
        }
        // put into v_ids
        v_ids.push_back(ids);
      }
      assert(!feof(fpF));
      std::string myBuf("");
      readMultiLine(myBuf, fpF);
      //std::istringstream is(myBuf);
      /*c = fgets(bufS, LENGTH_LINE_MAX, fpF);
      std::istringstream is(bufS);
                        */
      //buildChannelTargetsMap(maskVector, v_ids, is);
      buildChannelTargetsMap(maskVector, v_ids, myBuf);
      // memory clean v_ids
      for (std::vector<unsigned int*>::const_iterator it = v_ids.begin();
           it != v_ids.end(); it++)
      {
        delete *it;
      }
      v_ids.clear();
    }
    // delete[] ids;
  }
  else
    rval = false;

  _channels = rval;
  return _channels;
}

bool Params::readElectricalSynapseTargets(FILE* fpF)
{
  bool rval = true;
  _electricalSynapses = false;
  _electricalSynapseTargetsMask1 = _electricalSynapseTargetsMask2 = 0;
  _electricalSynapseTargetsMap.clear();
  int n = 0;
  char bufS[LENGTH_LINE_MAX];
  std::string tokS;
  jumpOverCommentLine(fpF);
  char* c = fgets(bufS, LENGTH_LINE_MAX, fpF);
  std::istringstream is(bufS);
  is >> tokS;
  if (tokS == "ELECTRICAL_SYNAPSE_TARGETS")
  {
    is >> n;
  }
  else
    rval = false;

  if (n > 0)
  {
    std::vector<SegmentDescriptor::SegmentKeyData> maskVector1, maskVector2;
    _electricalSynapseTargetsMask1 = resetMask(fpF, maskVector1);
    unsigned int sz1 = maskVector1.size();
    assert(sz1);
    _electricalSynapseTargetsMask2 = resetMask(fpF, maskVector2);
    unsigned int sz2 = maskVector2.size();
    assert(sz2);

    unsigned int* ids1 = new unsigned int[sz1];
    unsigned int* ids2 = new unsigned int[sz2];

    for (int i = 0; i < n; i++)  // for each line, not counting comment-line
    {
      jumpOverCommentLine(fpF);
      // one line:
      // 2 2     2 0   DenSpine [Voltage] 1.0
      for (unsigned int j = 0; j < sz1; ++j)
      {
        if (1 != fscanf(fpF, "%d", &ids1[j])) assert(0);
        if (maskVector1[j] == SegmentDescriptor::branchType)
          ids1[j] = ids1[j] - 1;  // make sure the BRANCHTYPE is 0-based
      }                           // read-in 2 2
      for (unsigned int j = 0; j < sz2; ++j)
      {
        if (1 != fscanf(fpF, "%d", &ids2[j])) assert(0);
        if (maskVector2[j] == SegmentDescriptor::branchType)
          ids2[j] = ids2[j] - 1;  // make sure the BRANCHTYPE is 0-based
      }                           // read-in 2 0
      assert(!feof(fpF));
      c = fgets(bufS, LENGTH_LINE_MAX, fpF);  // read-in DenSpine [Voltage] 1.0
      std::istringstream is(bufS);

      std::map<key_size_t, std::list<Params::ElectricalSynapseTarget> >&
          targetsMap =
              _electricalSynapseTargetsMap[_segmentDescriptor.getSegmentKey(
                  maskVector1, &ids1[0])];
      std::list<Params::ElectricalSynapseTarget>& targets =
          targetsMap[_segmentDescriptor.getSegmentKey(maskVector2, &ids2[0])];

      Params::ElectricalSynapseTarget st;
      st._parameter = -1.0;
      while (is >> st._type)
      {
        while (is.get() != '[')
        {
          assert(is.good());
        }
        char buf[LENGTH_IDNAME_MAX];
		assert(StringUtils::streamGet(is, buf, LENGTH_IDNAME_MAX, ']'));
        //is.get(buf, LENGTH_IDNAME_MAX, ']');
        std::string stringbuf(buf);
        std::vector<std::string> tokens;  // extract 'Voltage' as token
        StringUtils::Tokenize(stringbuf, tokens, " ,");
        for (std::vector<std::string>::iterator i = tokens.begin(),
                                                end = tokens.end();
             i != end; ++i)
        {
          st.addTarget(*i);
        }
        /*
                 char* tok = strtok(buf, " ,");
                 while (tok != 0) {
                 st.addTarget(std::string(tok));
                 tok = strtok(0, " ,");
                 }*/
        if (is.get() != ']') assert(0);
        is >> st._parameter;
        targets.push_back(st);
        st.clear();
      }
      targets.sort();
    }
    delete[] ids1;
    delete[] ids2;
  }
  else
    rval = false;
  _electricalSynapses = rval;
  return _electricalSynapses;
}

bool Params::readBidirectionalConnectionTargets(FILE* fpF)
{
	/*
BIDIRECTIONAL_CONNECTION_TARGETS 2
BRANCHTYPE MTYPE
BRANCHTYPE MTYPE 
3 2     3 0   DenSpine [Voltage Calcium CalciumER] 1.0
3 2     4 0   DenSpine [Voltage Calcium CalciumER] 1.0
	 */
  bool rval = true;
  _bidirectionalConnections = false;
  _bidirectionalConnectionTargetsMask1 = _bidirectionalConnectionTargetsMask2 =
      0;
  _bidirectionalConnectionTargetsMap.clear();
  int n = 0;
  char bufS[LENGTH_LINE_MAX];
  std::string tokS;
  jumpOverCommentLine(fpF);
  char* c = fgets(bufS, LENGTH_LINE_MAX, fpF);
  std::istringstream is(bufS);
  is >> tokS;
  if (tokS == "BIDIRECTIONAL_CONNECTION_TARGETS")
  {
    is >> n;
  }
  else
    rval = false;

  if (n > 0)
  {
    std::vector<SegmentDescriptor::SegmentKeyData> maskVector1, maskVector2;
    _bidirectionalConnectionTargetsMask1 = resetMask(fpF, maskVector1);
    unsigned int sz1 = maskVector1.size();
    assert(sz1);
    _bidirectionalConnectionTargetsMask2 = resetMask(fpF, maskVector2);
    unsigned int sz2 = maskVector2.size();
    assert(sz2);

    unsigned int* ids1 = new unsigned int[sz1];
    unsigned int* ids2 = new unsigned int[sz2];

    for (int i = 0; i < n; i++)  // for each line, not counting comment-line
    {
      jumpOverCommentLine(fpF);
      // one line:
      // 2 2     2 0   DenSpine [Voltage, Calcium] 1.0
      for (unsigned int j = 0; j < sz1; ++j)
      {
        if (1 != fscanf(fpF, "%d", &ids1[j])) assert(0);
        if (maskVector1[j] == SegmentDescriptor::branchType)
          ids1[j] = ids1[j] - 1;  // make sure the BRANCHTYPE is 0-based
      }                           // read-in 2 2
      for (unsigned int j = 0; j < sz2; ++j)
      {
        if (1 != fscanf(fpF, "%d", &ids2[j])) assert(0);
        if (maskVector2[j] == SegmentDescriptor::branchType)
          ids2[j] = ids2[j] - 1;  // make sure the BRANCHTYPE is 0-based
      }                           // read-in 2 0
      assert(!feof(fpF));
      c = fgets(bufS, LENGTH_LINE_MAX, fpF);  // read-in DenSpine [Voltage] 1.0
      std::istringstream is(bufS);

      std::map<key_size_t,
               std::list<Params::BidirectionalConnectionTarget> >& targetsMap =
          _bidirectionalConnectionTargetsMap[_segmentDescriptor.getSegmentKey(
              maskVector1, &ids1[0])];
      std::list<Params::BidirectionalConnectionTarget>& targets =
          targetsMap[_segmentDescriptor.getSegmentKey(maskVector2, &ids2[0])];

      Params::BidirectionalConnectionTarget st;
      st._parameter = -1.0;
      while (is >> st._type)
      {
        while (is.get() != '[')
        {
          assert(is.good());
        }
        char buf[LENGTH_IDNAME_MAX];
		assert(StringUtils::streamGet(is, buf, LENGTH_IDNAME_MAX, ']'));
        //is.get(buf, LENGTH_IDNAME_MAX, ']');
        std::string stringbuf(buf);
        std::vector<std::string> tokens;  // extract 'Voltage' as token
        StringUtils::Tokenize(stringbuf, tokens, " ,");
        for (std::vector<std::string>::iterator i = tokens.begin(),
                                                end = tokens.end();
             i != end; ++i)
        {
          st.addTarget(*i);
        }
        /*
                 char* tok = strtok(buf, " ,");
                 while (tok != 0) {
                 st.addTarget(std::string(tok));
                 tok = strtok(0, " ,");
                 }*/
        if (is.get() != ']') assert(0);
        is >> st._parameter; // probability of forming the spine-attachment
        targets.push_back(st);
        st.clear();
      }
      targets.sort();
    }
    delete[] ids1;
    delete[] ids2;
  }
  else
    rval = false;
  _bidirectionalConnections = rval;
  return _bidirectionalConnections;
}

bool Params::readChemicalSynapseTargets(FILE* fpF)
{
  /*
   * CHEMICAL_SYNAPSE_TARGETS 1
   * BRANCHTYPE MTYPE ETYPE
   * BRANCHTYPE MTYPE
   * 1 1 0   0 2   [AMPA NMDA] [Voltage] [Voltage] [Voltage] [Voltage, Calcium]
   * 1.0
   */
  bool rval = true;
  _chemicalSynapses = false;
  _chemicalSynapseTargetsMask1 = _chemicalSynapseTargetsMask2 = 0;
  _chemicalSynapseTargetsMap.clear();
  skipHeader(fpF);
  int n = 0;
  char bufS[LENGTH_LINE_MAX];
  std::string tokS;
  jumpOverCommentLine(fpF);
  char* c = fgets(bufS, LENGTH_LINE_MAX, fpF);
  std::istringstream is(bufS);
  is >> tokS;
  if (tokS == "CHEMICAL_SYNAPSE_TARGETS")
  {
    is >> n;
  }
  else
    rval = false;

  if (n > 0)
  {
    std::vector<SegmentDescriptor::SegmentKeyData> maskVector1, maskVector2;
    _chemicalSynapseTargetsMask1 = resetMask(fpF, maskVector1);
    unsigned int sz1 = maskVector1.size();
    assert(sz1);
    _chemicalSynapseTargetsMask2 = resetMask(fpF, maskVector2);
    unsigned int sz2 = maskVector2.size();
    assert(sz2);

    unsigned int* ids1 = new unsigned int[sz1];
    unsigned int* ids2 = new unsigned int[sz2];

    for (int i = 0; i < n; i++)
    {
      jumpOverCommentLine(fpF);
      for (unsigned int j = 0; j < sz1; ++j)
      {
        if (1 != fscanf(fpF, "%d", &ids1[j])) assert(0);
        if (maskVector1[j] == SegmentDescriptor::branchType)
          ids1[j] = ids1[j] - 1;  // make sure the BRANCHTYPE is 0-based
      }
      for (unsigned int j = 0; j < sz2; ++j)
      {
        if (1 != fscanf(fpF, "%d", &ids2[j])) assert(0);
        if (maskVector2[j] == SegmentDescriptor::branchType)
          ids2[j] = ids2[j] - 1;  // make sure the BRANCHTYPE is 0-based
      }
      assert(!feof(fpF));
      c = fgets(bufS, LENGTH_LINE_MAX, fpF);
      std::istringstream is(bufS);

      std::map<key_size_t, std::list<Params::ChemicalSynapseTarget> >&
          targetsMap =
              _chemicalSynapseTargetsMap[_segmentDescriptor.getSegmentKey(
                  maskVector1, &ids1[0])];
      std::list<Params::ChemicalSynapseTarget>& targets =
          targetsMap[_segmentDescriptor.getSegmentKey(maskVector2, &ids2[0])];

      Params::ChemicalSynapseTarget st;
      st._parameter = -1.0;

      std::vector<std::string> types;

      while (is.get() != '[')
      {
        assert(is.good());
      }
      char buf1[LENGTH_IDNAME_MAX];
	  assert(StringUtils::streamGet(is, buf1, LENGTH_IDNAME_MAX, ']'));
      //is.get(buf1, LENGTH_IDNAME_MAX, ']');
      std::string stringbuf(buf1);
      std::vector<std::string> tokens;
      StringUtils::Tokenize(stringbuf, tokens, " ,");
      for (std::vector<std::string>::iterator ii = tokens.begin(),
                                              end = tokens.end();
           ii != end; ++ii)
      {
        types.push_back(*ii);
      }
      /*
      char* tok = strtok(buf1, " ,");
      while (tok != 0) {
          types.push_back(std::string(tok));
          tok = strtok(0, " ,");
      }
      */
      if (is.get() != ']') assert(0);

      for (std::vector<int>::size_type ii = 0; ii < types.size(); ++ii)
      {
        while (is.get() != '[')
        {
          assert(is.good());
        }
        char buf1[LENGTH_IDNAME_MAX];
		assert(StringUtils::streamGet(is, buf1, LENGTH_IDNAME_MAX, ']'));
        //is.get(buf1, LENGTH_IDNAME_MAX, ']');
        std::string stringbuf(buf1);
        std::vector<std::string> tokens;
        StringUtils::Tokenize(stringbuf, tokens, " ,");
        for (std::vector<std::string>::iterator j = tokens.begin(),
                                                end = tokens.end();
             j != end; ++j)
        {
          st.addTarget1(types[ii], *j);
        }
        /*
        char* tok1 = strtok(buf1, " ,");
        while (tok1 != 0) {
            st.addTarget1(types[ii], std::string(tok1));
            tok1 = strtok(0, " ,");
        }*/
        if (is.get() != ']') assert(0);
        while (is.get() != '[')
        {
          assert(is.good());
        }
        char buf2[LENGTH_IDNAME_MAX];
		assert(StringUtils::streamGet(is, buf2, LENGTH_IDNAME_MAX, ']'));
        //is.get(buf2, LENGTH_IDNAME_MAX, ']');
        if (is.get() != ']') assert(0);
        stringbuf = std::string(buf2);
        // std::vector<std::string> tokens;
        StringUtils::Tokenize(stringbuf, tokens, " ,");
        for (std::vector<std::string>::iterator j = tokens.begin(),
                                                end = tokens.end();
             j != end; ++j)
        {
          st.addTarget2(types[ii], *j);
        }
        /*
        char* tok2 = strtok(buf2, " ,");
        while (tok2 != 0) {
            st.addTarget2(types[ii], std::string(tok2));
            tok2 = strtok(0, " ,");
        }*/
      }
      is >> st._parameter;
      targets.push_back(st);
      st.clear();
      targets.sort();
    }
    delete[] ids1;
    delete[] ids2;
  }
  else
    rval = false;
  _chemicalSynapses = rval;
  return _chemicalSynapses;
}

bool Params::readPreSynapticPointTargets(FILE* fpF)
{
  /* Example:
   * PRESYNAPTIC_POINT_TARGETS 3
   * AMPA Voltage
   * NMDA Voltage
   * GABAA Voltage
   */
  bool rval = true;
  _preSynapticPointTargetsMap.clear();
  _preSynapticPointSynapseMap.clear();
  skipHeader(fpF);
  int n = 0;
  char bufS[LENGTH_LINE_MAX], tokS[LENGTH_TOKEN_MAX], tokS2[LENGTH_TOKEN_MAX];
  jumpOverCommentLine(fpF);
  char* c = fgets(bufS, LENGTH_LINE_MAX, fpF);
  if (2 == sscanf(bufS, "%s %d ", tokS, &n))
  {
    std::string btype(tokS);
    std::string expected_btype("PRESYNAPTIC_POINT_TARGETS");
    if (btype == expected_btype)
    {
      // do nothing
    }
    else
      rval = false;
  }
  else
    rval = false;
  if (n > 0)
  {
    for (int i = 0; i < n; i++)  // for each line, not counting comment-line
    {
      jumpOverCommentLine(fpF);
      c = fgets(bufS, LENGTH_LINE_MAX, fpF);
      if (2 == sscanf(bufS, "%s %s ", tokS, tokS2))
      {
        std::string synID(tokS);
        std::string targetID(tokS2);
        _preSynapticPointTargetsMap[synID] = targetID;
        _preSynapticPointSynapseMap[targetID].push_back(synID);
      }
      else
      {
        rval = false;
        assert(0);
      }
    }
  }
  else
    rval = false;
  return rval;
}

void Params::skipHeader(FILE* fpF)
{
  int pos = ftell(fpF);
  char bufS[LENGTH_LINE_MAX];
  do
  {
    pos = ftell(fpF);
    char* c = fgets(bufS, LENGTH_LINE_MAX, fpF);
  } while (bufS[0] == '#');
  fseek(fpF, pos, SEEK_SET);
}

unsigned long long Params::readNamedParam(
    FILE* fpF, std::string name, std::map<key_size_t, double>& namedParamsMap)
{
  unsigned long long mask = 0;
  bool rval = true;
  namedParamsMap.clear();
  int n = 0;
  char bufS[LENGTH_LINE_MAX], tokS[LENGTH_TOKEN_MAX];
  jumpOverCommentLine(fpF);
  char* c = fgets(bufS, LENGTH_LINE_MAX, fpF);
  if (2 == sscanf(bufS, "%s %d ", tokS, &n))
  {
    std::string btype(tokS);
    if (btype == name)
    {
      // do nothing
    }
    else
      rval = false;
  }
  else
    rval = false;

  assert(n > 0);

  std::vector<SegmentDescriptor::SegmentKeyData> maskVector;
  mask = resetMask(fpF, maskVector);
  double p;
  unsigned int sz = maskVector.size();
  assert(sz);
  unsigned int* ids = new unsigned int[sz];

  for (int i = 0; i < n; i++)  // for each line, not counting comment-line
  {
    jumpOverCommentLine(fpF);
    for (int j = 0; j < sz; ++j)
    {
      if (1 != fscanf(fpF, "%d", &ids[j])) assert(0);
      if (maskVector[j] == SegmentDescriptor::branchType)
        ids[j] = ids[j] - 1;  // make sure the BRANCHTYPE is 0-based
    }
    assert(!feof(fpF));
    c = fgets(bufS, LENGTH_LINE_MAX, fpF);
    sscanf(bufS, "%lf", &p);
    namedParamsMap[_segmentDescriptor.getSegmentKey(maskVector, &ids[0])] = p;
  }
  delete[] ids;
  assert(rval);
  return mask;
}

bool Params::readCompartmentVariableCosts(FILE* fpF)
{
  bool rval = true;
  _compartmentVariableCostsMap.clear();
  int n = 0;
  char bufS[LENGTH_LINE_MAX], tokS[LENGTH_TOKEN_MAX];
  jumpOverCommentLine(fpF);
  char* c = fgets(bufS, LENGTH_LINE_MAX, fpF);
  if (2 == sscanf(bufS, "%s %d ", tokS, &n))
  {
    std::string btype(tokS);
    std::string expected_btype("COMPARTMENT_VARIABLE_COSTS");
    if (btype == expected_btype)
    {
      // do nothing
    }
    else
      rval = false;
  }
  else
    rval = false;

  if (n > 0)
  {
    double cost;
    for (int i = 0; i < n; i++)  // for each line, not counting comment-line
    {
      jumpOverCommentLine(fpF);
      char* c = fgets(bufS, LENGTH_LINE_MAX, fpF);
      if (2 == sscanf(bufS, "%s %lf ", tokS, &cost))
      {
        std::string chanID(tokS);
        _compartmentVariableCostsMap[chanID] = cost;
      }
      else
      {
        rval = false;
        assert(0);
      }
    }
  }
  else
    rval = false;
  return rval;
}

bool Params::readChannelCosts(FILE* fpF)
{
  bool rval = true;
  _channelCostsMap.clear();
  int n = 0;
  char bufS[LENGTH_LINE_MAX], tokS[LENGTH_TOKEN_MAX];
  jumpOverCommentLine(fpF);
  char* c = fgets(bufS, LENGTH_LINE_MAX, fpF);
  if (2 == sscanf(bufS, "%s %d ", tokS, &n))
  {
    std::string btype(tokS);
    std::string expected_btype("CHANNEL_COSTS");
    if (btype == expected_btype)
    {
      // do nothing
    }
    else
      rval = false;
  }
  else
    rval = false;

  if (n > 0)
  {
    double cost;
    for (int i = 0; i < n; i++)  // for each line, not counting comment-line
    {
      jumpOverCommentLine(fpF);
      c = fgets(bufS, LENGTH_LINE_MAX, fpF);
      if (2 == sscanf(bufS, "%s %lf ", tokS, &cost))
      {
        std::string chanID(tokS);
        _channelCostsMap[chanID] = cost;
      }
      else
      {
        rval = false;
        assert(0);
      }
    }
  }
  else
    rval = false;
  return rval;
}

bool Params::readModelParams(
    FILE* fpF, const std::string& id,
    std::map<std::string, unsigned long long>& paramsMasks,
    std::map<
        std::string,
        std::map<key_size_t, std::list<std::pair<std::string, dyn_var_t> > > >&
        paramsMap,
    std::map<
        std::string,
        std::map<key_size_t, std::list<std::pair<std::string,
                                                 std::vector<dyn_var_t> > > > >&
        arrayParamsMap)
{
  /* Example of input from fpF
	 * NOTE: id = COMPARTMENT_VARIABLE_PARAMS
   * Here, there are 2 sub-groups
COMPARTMENT_VARIABLE_PARAMS 2
Voltage 3
MTYPE BRANCHTYPE
0 1 <Cm=0.01>
0 1 <gLeak=0.000338>
0 2 <Cm=0.01>
Calcium 3
MTYPE BRANCHTYPE
0 1 <CaClearance=1.1>
0 3 <CaClearance=4.2>
0 4 <CaClearance=4.2>
Cah 8
BRANCHTYPE MTYPE
1 0 <gbar={0.00992}>
3 0 <gbar={0.00992}>
4 0 <gbar_dists={380.0,480.0}>
4 0 <gbar_values={0.000555,0.00555,0.000555}>
1 2 <gbar={0.00992}>
   */
  bool rval = true;
  paramsMasks.clear();
  paramsMap.clear();
  int n = 0;
  char bufS[LENGTH_LINE_MAX], tokS[LENGTH_TOKEN_MAX];
  jumpOverCommentLine(fpF);
  char* c = fgets(bufS, LENGTH_LINE_MAX, fpF);
  if (2 == sscanf(bufS, "%s %d ", tokS, &n))
  {  // find number of subgroups
		//read line:   COMPARTMENT_VARIABLE_PARAMS 2
    std::string btype(tokS);
    std::string expected_btype(id);
    if (btype == expected_btype)
    {
      // do nothing
    }
    else
      rval = false;
  }
  else
    rval = false;

  if (n > 0)
  {
    for (int i = 0; i < n; i++)  // for each subgroup
    {
      /* Two examples:
			 * One group for compartment
Calcium 3
MTYPE BRANCHTYPE
0 1 <CaClearance=1.1>
0 3 <CaClearance=4.2>
0 4 <CaClearance=4.2; other=5>

        One group for channel
Cah 8
BRANCHTYPE MTYPE
1 0 <gbar={0.00992}>
3 0 <gbar={0.00992}; other = 5>
4 0 <gbar_dists={380.0,480.0}>
4 0 <gbar_values={0.000555,0.00555,0.000555}>
1 2 <gbar={0.00992}>
       */
      jumpOverCommentLine(fpF);
      int p;
      c = fgets(bufS, LENGTH_LINE_MAX, fpF);
      if (2 == sscanf(bufS, "%s %d ", tokS, &p))  // e.g.: Calcium 3
      {
        std::string modelID(tokS);
        std::vector<SegmentDescriptor::SegmentKeyData> maskVector;
        paramsMasks[modelID] = resetMask(fpF, maskVector);
        unsigned int sz = maskVector.size();
        assert(sz);
				for (unsigned int k = 0; k < sz; ++k)
				{  // validate
					if (maskVector[k] == SegmentDescriptor::segmentIndex)
					{
						std::cerr << "Params : Targeting channel parameters to "
							"individual compartments not supported!"
							<< std::endl;
						return false;
						//exit(EXIT_FAILURE);
					}
				}	

        //unsigned int* ids = new unsigned int[sz]();
				for (int j = 0; j < p;
						j++)  // for each line (subgroup), not counting comment-line
				{
					jumpOverCommentLine(fpF);
					std::vector<unsigned int*> v_ids;
					if (checkForSpecialCases(fpF, sz))
					{
						// check if a special case is used
						// A special case can be either:
						//      range [2, 5]
						//      range [2:5]
						//      range [2,5:7]
						// then, create a vector of v_ids
						// NOTE: We planed to support
						//      asterisk *
						//  but then it requires implicit knowledge of the range of value
						//  so we don't support it now

						unsigned int* ids = new unsigned int[sz];
						std::vector<std::vector<int> > vect_values;
						int total_vect = 1;
						for (unsigned int j = 0; j < sz; ++j)
						{
							std::vector<int> values;
							int val = 0;
							// LIMIT: current only support  special case for BRANCHTYPE
							if (maskVector[j] == SegmentDescriptor::branchType)
							{
								getListofValues(fpF, values);  // assume the next data to read is in
								// the form  [...] and it occurs for
								// BRANCHTYPE
								/*  if (isAsterisk(fpF))
										{
										assert(0);
										}
										else if (isBracket(fpF, lval, hval))
										{
										}
										*/
								// make sure the BRANCHTYPE is 0-based
								for (std::vector<int>::iterator it = values.begin();
										it != values.end(); ++it)
								{
									*it -= 1;
								}
								// std::for_each(values.begin(), values.end(), [](int & d) { d-=1;
								// });
							}
							else
							{
								if (1 != fscanf(fpF, "%d", &val)) assert(0);
								values.push_back(val);
							}
							/*
								 if (1 != fscanf(fpF, "%d", &ids[j])) assert(0);
								 if (maskVector[j] == SegmentDescriptor::branchType)
								 ids[j] = ids[j] - 1;  // make sure the BRANCHTYPE is 0-based
								 */
							vect_values.push_back(values);
							total_vect *= values.size();
						}
						// generate all array elements in the vector
						{
							unsigned int** pids = new unsigned int* [total_vect];
							for (int jj = 0; jj < total_vect; ++jj)
							{
								pids[jj] = new unsigned int[sz]();
								v_ids.push_back(pids[jj]);
							}

							// fill the data
							for (unsigned int jj = 0; jj < sz; jj++)
							{
								int num2clone = 1;
								for (unsigned int xx = jj + 1; xx < sz; xx++)
									num2clone *= vect_values[xx].size();
								int gap = num2clone * vect_values[jj].size();

								for (unsigned int kk = 0; kk < vect_values[jj].size(); kk++)
								{
									for (int xx = (num2clone) * (kk); xx < total_vect; xx += gap)
									{
										std::vector<unsigned int*>::iterator iter,
											iterstart = v_ids.begin() + xx,
											iterend = v_ids.begin() + xx + num2clone - 1;
										for (iter = iterstart; iter <= iterend; iter++)
											(*iter)[jj] = vect_values[jj][kk];
									}
								}
							}
						}
					}
					else
					{
						// v_ids[0] = new unsigned int [sz];
						// unsigned int* ids = v_ids[0];
						unsigned int* ids = new unsigned int[sz]();
						for (unsigned int kk = 0; kk < sz; ++kk)
						{// read vector mask part
							if (1 != fscanf(fpF, "%d", &ids[kk])) assert(0);
							if (maskVector[kk] == SegmentDescriptor::branchType)
							{
								ids[kk] = ids[kk] - 1;  // make sure the BRANCHTYPE is 0-based
							}
						}
						// put into v_ids
						v_ids.push_back(ids);
					}
          assert(!feof(fpF));

					std::string myBuf("");
					readMultiLine(myBuf, fpF);
					/*c = fgets(bufS, LENGTH_LINE_MAX, fpF);
          std::istringstream is(bufS);*/
          //buildParamsMap(maskVector, v_ids, is, modelID, paramsMap, arrayParamsMap);
          buildParamsMap(maskVector, v_ids, myBuf, modelID, paramsMap, arrayParamsMap);
					// memory clean v_ids
					for (std::vector<unsigned int*>::const_iterator it = v_ids.begin();
							it != v_ids.end(); it++)
					{
						delete *it;
					}
					v_ids.clear();
				}
				//delete[] ids;
      }
      else
      {
        rval = false;
        assert(0);
      }
    }
  }
  else
    rval = false;
  return rval;
}

bool Params::readElectricalSynapseCosts(FILE* fpF)
{
  bool rval = true;
  _electricalSynapseCostsMap.clear();
  int n = 0;
  char bufS[LENGTH_LINE_MAX], tokS[LENGTH_TOKEN_MAX];
  jumpOverCommentLine(fpF);
  char* c = fgets(bufS, LENGTH_LINE_MAX, fpF);
  if (2 == sscanf(bufS, "%s %d ", tokS, &n))
  {
    std::string btype(tokS);
    std::string expected_btype("ELECTRICAL_SYNAPSE_COSTS");
    if (btype == expected_btype)
    {
      // do nothing
    }
    else
      rval = false;
  }
  else
    rval = false;

  if (n > 0)
  {
    double cost;
    for (int i = 0; i < n; i++)  // for each line, not counting comment-line
    {
      jumpOverCommentLine(fpF);
      c = fgets(bufS, LENGTH_LINE_MAX, fpF);
      if (2 == sscanf(bufS, "%s %lf ", tokS, &cost))
      {
        std::string synID(tokS);
        _electricalSynapseCostsMap[synID] = cost;
      }
      else
      {
        rval = false;
        assert(0);
      }
    }
  }
  else
    rval = false;
  return rval;
}

bool Params::readBidirectionalConnectionCosts(FILE* fpF)
{
  bool rval = true;
  _bidirectionalConnectionCostsMap.clear();
  int n = 0;
  char bufS[LENGTH_LINE_MAX], tokS[LENGTH_TOKEN_MAX];
  jumpOverCommentLine(fpF);
  char* c = fgets(bufS, LENGTH_LINE_MAX, fpF);
  if (2 == sscanf(bufS, "%s %d ", tokS, &n))
  {
    std::string btype(tokS);
    std::string expected_btype("BIDIRECTIONAL_CONNECTION_COSTS");
    if (btype == expected_btype)
    {
      // do nothing
    }
    else
      rval = false;
  }
  else
    rval = false;

  if (n > 0)
  {
    double cost;
    for (int i = 0; i < n; i++)  // for each line, not counting comment-line
    {
      jumpOverCommentLine(fpF);
      c = fgets(bufS, LENGTH_LINE_MAX, fpF);
      if (2 == sscanf(bufS, "%s %lf ", tokS, &cost))
      {
        std::string synID(tokS);
        _bidirectionalConnectionCostsMap[synID] = cost;
      }
      else
      {
        rval = false;
        assert(0);
      }
    }
  }
  else
    rval = false;
  return rval;
}
bool Params::readChemicalSynapseCosts(FILE* fpF)
{
  bool rval = true;
  _chemicalSynapseCostsMap.clear();
  int n = 0;
  char bufS[LENGTH_LINE_MAX], tokS[LENGTH_TOKEN_MAX];
  jumpOverCommentLine(fpF);
  char* c = fgets(bufS, LENGTH_LINE_MAX, fpF);
  if (2 == sscanf(bufS, "%s %d ", tokS, &n))
  {
    std::string btype(tokS);
    std::string expected_btype("CHEMICAL_SYNAPSE_COSTS");
    if (btype == expected_btype)
    {
      // do nothing
    }
    else
      rval = false;
  }
  else
    rval = false;
  if (n > 0 and rval)
  {
    double cost;
    for (int i = 0; i < n; i++)  // for each line, not counting comment-line
    {
      jumpOverCommentLine(fpF);
      c = fgets(bufS, LENGTH_LINE_MAX, fpF);
      if (2 == sscanf(bufS, "%s %lf ", tokS, &cost))
      {
        std::string synID(tokS);
        _chemicalSynapseCostsMap[synID] = cost;
      }
      else
      {
        rval = false;
        assert(0);
      }
    }
    rval = true;
  }
  else
    rval = false;
  return rval;
}

unsigned long long Params::resetMask(
    FILE* fpF, std::vector<SegmentDescriptor::SegmentKeyData>& maskVector)
{
  /* read on the line containing the name(s) of one or more field name in the
   * key,
   * separated by spaces
   * e.g.: BRANCHTYPE MTYPE
   * check SegmentDescriptor class
   */
  maskVector.clear();
  char bufS[LENGTH_LINE_MAX];
  jumpOverCommentLine(fpF);
  char* c = fgets(bufS, LENGTH_LINE_MAX, fpF);
  std::istringstream is(bufS);
  std::string tokS;
  is >> tokS;
  while (!is.eof())
  {
    maskVector.push_back(_segmentDescriptor.getSegmentKeyData(tokS));
    is >> tokS;
  }
  return _segmentDescriptor.getMask(maskVector);
}
// INPUT:
//  fpF  = file pointer (FILE*) to the opening parameter file
// OUTPUT:
//  maskVector = a vector containing the indices of all given fieldname
//  bufS       = containing the read line
// GOAL:
//  read the next line from 'fpF' which is expected to contain the
//  space-delimited
//  list of fieldnames
//  Check SegmentDescriptor.h for the list of defined fieldnames
unsigned long long Params::resetMask(
    FILE* fpF, std::vector<SegmentDescriptor::SegmentKeyData>& maskVector,
    char* bufS)
{
  /* read on the line containing the name(s) of one or more field name in the
   * key,
   * separated by spaces
   * e.g.: BRANCHTYPE MTYPE
   * check SegmentDescriptor class
   */
  maskVector.clear();
  jumpOverCommentLine(fpF);
  char* c = fgets(bufS, LENGTH_LINE_MAX, fpF);
  std::istringstream is(bufS);
  std::string tokS;
  is >> tokS;
  while (!is.eof())
  {
    maskVector.push_back(_segmentDescriptor.getSegmentKeyData(tokS));
    is >> tokS;
  }
  return _segmentDescriptor.getMask(maskVector);
}

// GOAL: read until the end of the line
//    if the last non-space character is '\'
//    then continue to the next line
// NOTE: existing content of 'out_bufS'  is cleared before loading data
void Params::readMultiLine(std::string& out_bufS, FILE* fpF)
{
  char bufS[LENGTH_LINE_MAX];
	out_bufS.clear();
  //NOTE: The '\n' new-line character is read-in as well
  char* c = fgets(bufS, sizeof(bufS), fpF);
  bool itsok = true;
  do
  {
    int i = 1;
    while (strlen(bufS) >= i and
           (bufS[strlen(bufS) - i] == ' ' || bufS[strlen(bufS) - i] == '\n'))
    {//move index to the first non-space character on the right-side of buffer
      i++;
    }
    if (strlen(bufS) >= i)
    {//using 'i' to check if backslash is used
      std::string tmp(bufS);
      if (bufS[strlen(bufS) - i] == '\\')
      {//automatically ignore the last backslash character
				out_bufS.append(tmp.substr(0, strlen(bufS) - i)); 
				jumpOverCommentLine(fpF);
        c = fgets(bufS, sizeof(bufS), fpF);
      }
      else
			{
				out_bufS.append(tmp.substr(0, strlen(bufS) - i+1)); 
        itsok = false;
      }
    }
  } while (itsok);
}

// Given a vector v_ids of matched pattern
// Given a stream of channel target
// GOAL: create _channelTargetsMap
void Params::buildChannelTargetsMap(
    std::vector<SegmentDescriptor::SegmentKeyData>& maskVector,
    std::vector<unsigned int*>& v_ids, 
		//std::istringstream& is
		const std::string & myBuf
		)
{
  std::vector<unsigned int*>::const_iterator iter = v_ids.begin(),
                                             iterend = v_ids.end();
  for (; iter < iterend; iter++)
  {
    unsigned int* ids = *iter;
    std::list<Params::ChannelTarget>& targets =
        _channelTargetsMap[_segmentDescriptor.getSegmentKey(maskVector,
                                                            &ids[0])];
    Params::ChannelTarget ct;
		std::istringstream is(myBuf);
    while (is >> ct._type)
    {
      while (is.get() != '[')
      {
        assert(is.good());
      }
      char buf1[LENGTH_IDNAME_MAX];
	  assert(StringUtils::streamGet(is, buf1, LENGTH_IDNAME_MAX, ']'));
      //is.get(buf1, LENGTH_IDNAME_MAX, ']');

      std::string stringbuf(buf1);
      std::vector<std::string> tokens;
      StringUtils::Tokenize(stringbuf, tokens, " ,");
      for (std::vector<std::string>::iterator i = tokens.begin();
           i != tokens.end(); ++i)
      {
        ct.addTarget1(*i);
      }
      /*char* tok1 = strtok(buf1, " ,");
              while (tok1 != 0) {
              ct.addTarget1(std::string(tok1));
              tok1 = strtok(0, " ,");
              }*/
      if (is.get() != ']') assert(0);
      while (is.get() != '[')
      {
        assert(is.good());
      }
      char buf2[LENGTH_IDNAME_MAX];
	  assert(StringUtils::streamGet(is, buf2, LENGTH_IDNAME_MAX, ']'));
      //is.get(buf2, LENGTH_IDNAME_MAX, ']');
      if (is.get() != ']') assert(0);
      stringbuf = std::string(buf2);
      StringUtils::Tokenize(stringbuf, tokens, " ,");
      for (std::vector<std::string>::iterator i = tokens.begin();
           i != tokens.end(); ++i)
      {
        ct.addTarget2(*i);
      }
      /*
               char* tok2 = strtok(buf2, " ,");
               while (tok2 != 0) {
               ct.addTarget2(std::string(tok2));
               tok2 = strtok(0, " ,");
               }
               */
      targets.push_back(ct);
      ct.clear();
    }
    targets.sort();
  }
}

// Given a vector v_ids of matched pattern
// Given a stream of channel's parameters
//            or compartment variable's parameters
// GOAL: create _...ParamsMap
void Params::buildParamsMap(
    std::vector<SegmentDescriptor::SegmentKeyData>& maskVector,
    std::vector<unsigned int*>& v_ids, 
		//std::istringstream& is_origin, 
		const std::string& myBuf, 
		std::string & modelID,
		std::map<std::string,
		std::map<key_size_t, std::list<std::pair<std::string, dyn_var_t> > > >&
		paramsMap,
		std::map<
		std::string,
		std::map<key_size_t,
		std::list<std::pair<std::string, std::vector<dyn_var_t> > > > >&
		arrayParamsMap
		)
{
  std::vector<unsigned int*>::const_iterator iter = v_ids.begin(),
                                             iterend = v_ids.end();
	for (; iter < iterend; iter++)
	{//for each element in v_ids, apply the same read-in data to it
		unsigned int* ids = *iter;
		std::list<std::pair<std::string, dyn_var_t> >& params =
			paramsMap[modelID][_segmentDescriptor.getSegmentKey(maskVector,
					&ids[0])];
		std::list<std::pair<std::string, std::vector<dyn_var_t> > >&
			arrayParams =
			arrayParamsMap[modelID][_segmentDescriptor.getSegmentKey(
					maskVector, &ids[0])];
		/* Support form of data
NOTE: must starts with '<' and ends with'>'
<gbar={0.0343}> //single name=val
<gbar=0.0343>
<gbar_dists={380.0,480.0}>  // one name-multiple-values
<gbar_values={0.00187,0.187,0.00187}>
<Cm=0.01; gLeak=0.000325> // multiple name=val
		 */
		std::istringstream is(myBuf);
		while (is.get() != '<')
		{
			assert(is.good());
		}
		char buf1[LENGTH_LINE_MAX];
		assert(StringUtils::streamGet(is, buf1, LENGTH_LINE_MAX, '>'));
		//is.get(buf1, LENGTH_LINE_MAX, '>');
		/*  buf1 looks like any of these
				<gbar={24.4}>
				<gbar={24.4}; other=5>
				<gbar_dists={24.4, 4.6}; other=5>
				*/
		std::string stringbuf1(buf1);  // to replace the code below
		std::vector<std::string> tokens1;
		StringUtils::Tokenize(stringbuf1, tokens1, ";");
		for (std::vector<std::string>::iterator ii = tokens1.begin(),
				end1 = tokens1.end();
				ii != end1; ++ii)
		{
			std::string delimiter = "=";
			size_t pos = (*ii).find(delimiter);
			std::string name = (*ii).substr(0, pos);
			StringUtils::trim(name);
			(*ii).erase(0, pos + delimiter.length());

			delimiter = " =";
			pos = (*ii).find(delimiter);
			std::string tok2 = (*ii).substr(0, pos);

			std::istringstream is2(tok2);
			if (is2.get() != '{')
			{  // single value
				dyn_var_t value = atof(tok2.c_str());
				params.push_back(std::pair<std::string, dyn_var_t>(name, value));
			}
			else
			{  // contain multiple values (comma-separated)
				std::vector<dyn_var_t> value;
				char buf2[LENGTH_LINE_MAX];
				/* NOTE: This code is potentialy bug when token info is too long
				is2.get(buf2, LENGTH_IDNAME_MAX, '}');
				*/
				assert(StringUtils::streamGet(is2, buf2, LENGTH_LINE_MAX, '}'));
				std::string stringbuf(buf2);
				std::vector<std::string> tokens;
				StringUtils::Tokenize(stringbuf, tokens, ",");
				for (std::vector<std::string>::iterator jj = tokens.begin(),
						end = tokens.end();
						jj != end; ++jj)
				{
					// assume input values are numerics
					value.push_back(atof((*jj).c_str()));
				}
				arrayParams.push_back(
						std::pair<std::string, std::vector<dyn_var_t> >(name, value));
			}
		}
		/*
			 char* tok1 = strtok(buf1, ";");
			 while (tok1 != 0) {
			 char* tok2 = strtok(tok1, "=");
			 std::string name(tok2);
			 tok2 = strtok(0, " =");
			 std::istringstream is2(tok2);
			 if (is2.get() != '{') {
			 dyn_var_t value = atof(tok2);
			 params.push_back(std::pair<std::string, dyn_var_t>(name,
			 value));
			 } else {
			 std::vector<dyn_var_t> value;
			 char buf2[LENGTH_IDNAME_MAX];
			 is2.get(buf2, LENGTH_IDNAME_MAX, '}');
			 char* tok3 = strtok(buf2, ",");
			 while (tok3 != 0) {
			 value.push_back(atof(tok3));
			 tok3 = strtok(0, ",");
			 }
			 arrayParams.push_back(
			 std::pair<std::string, std::vector<dyn_var_t> >(name,
			 value));
			 }
			 tok1 = strtok(0, ";");
			 }*/
	}

}

// Given a vector v_ids whose element is a vector of matched pattern
// Given a string of compartments' names
// GOAL: create _compartmentVariableTargetsMap
void Params::buildCompartmentVariableTargetsMap(
    std::vector<SegmentDescriptor::SegmentKeyData>& maskVector,
    std::vector<unsigned int*>& v_ids, 
		const std::string &myBuf
		//std::istringstream& is
		)
{
  std::vector<unsigned int*>::const_iterator iter = v_ids.begin(),
                                             iterend = v_ids.end();
	for (; iter < iterend; iter++)
	{
		unsigned int* ids = *iter;
		std::list<std::string> targets;
		std::string type;
		std::istringstream is(myBuf);
		while (is >> type)
		{
			targets.push_back(type);
		}
		targets.sort();
		_compartmentVariableTargetsMap[_segmentDescriptor.getSegmentKey(
				maskVector, &ids[0])] = targets;
	}
}

// GOAL: check if the next 'sz' space-separated words
//     are all numerics or not
//  true = all are numerics
//  SCENARIO:   sz = 4
//  1 4 5 6  --> true
//  1 [4,6] 5 6 --> false
//  * 4 5 6  --> false
bool Params::checkForSpecialCases(FILE* fpF, int sz)
{
  bool rval = false;
  fpos_t fpos;
  fgetpos(fpF, &fpos);
  int ival;
	char oneword[LENGTH_TOKEN_MAX];
  for (unsigned int j = 0; j < sz; ++j)
  {
		char c = fscanf(fpF, " %s", oneword);
    if (oneword[0] == '[' || oneword[0] == '*')
    {
      rval = true;
      break;
    }
  }
  fsetpos(fpF, &fpos);
  return rval;
}

// SCENARIO: the next character in file pointed by fpF
//   should be a string in such forms
//   [4:6]
//   [2, 4:6]
// GOAL: extract and return the vectors containing all values
//   4,5,6
//   2,4,5,6
void Params::getListofValues(FILE* fpF, std::vector<int>& values)
{
  char ch;
	bool readOK = true;
	values.clear();//reset data first

  while ((ch = fgetc(fpF)) != '[')
  {
  }
	if (feof(fpF))
	{
		std::cerr << "Syntax error of parameter file: expect range, e.g. [val1,val2] or [val1:val2]" << std::endl;
		exit(-1);
	}
  ch = fgetc(fpF);//read character after '['
  while (ch != ']' and readOK and !feof(fpF))
  {
    //char str[LENGTH_TOKEN_MAX];
		std::string str("");
    //int i = 0;
    do
    {
      //str[i] = ch;
			str.push_back(ch);
      //i++;
      ch = fgetc(fpF);
    } while (ch != ',' and ch != ']' and ch != ':' and !feof(fpF));
		if (feof(fpF))
		{
			readOK = false;
			break;
		}
    //str[i] = '\0';
		
    if (ch == ',')
    {
      values.push_back(atoi(str.c_str()));
    }
    else if (ch == ':')
    {
      int val1 = atoi(str.c_str());
      ch = fgetc(fpF);
      //i = 0;
			str.clear();
      do
      {
        //str[i] = ch;
        //i++;
			  str.push_back(ch);
        ch = fgetc(fpF);
        if (ch == ':') assert(0);//we cann't have two ':' at the same time, e.g. 2:3:5
      } while (ch != ',' and ch != ']' and !feof(fpF));
      //str[i] = '\0';

      int val2 = atoi(str.c_str());
      for (int i = val1; i <= val2; i++) values.push_back(i);
    }
    else if (ch == ']')
		{
			if (str.length() > 0)
				values.push_back(atoi(str.c_str()));
      break;
		}
		if (ch != ']')
			ch = fgetc(fpF);
  }
	if (!readOK)
	{
		std::cerr << "Syntax error of parameter file: expect range, e.g. [val1,val2] or [val1:val2]" << std::endl;
		exit(-1);
	}
}

///NOTE: Accept traditional ** array
void Params::readMarkovModel(const std::string& fname, dyn_var_t** &matChannelRateConstant,
		int &numChanStates, int* &vChannelStates, int &initialstate)
{
  FILE* fpF = fopen(fname.c_str(), "r");
  char bufS[LENGTH_LINE_MAX];
	if (fpF == NULL)
	{
		std::cerr << "File " << fname << " not found.\n";
		assert(fpF);
	}
	//read in the number of states
	jumpOverCommentLine(fpF);
  char* c = fgets(bufS, LENGTH_LINE_MAX, fpF);
  if (1 != sscanf(bufS, "%d ", &numChanStates))
	{
    std::cerr << "Syntax of Markov-model file invalid: expect \n #-states" << std::endl;
		exit(EXIT_FAILURE);
	}

	//read in which state(s) is open-state
	jumpOverCommentLine(fpF);
  vChannelStates =	new int[numChanStates]();
  c = fgets(bufS, LENGTH_LINE_MAX, fpF);
	std::string str(bufS);
	std::vector<std::string> tokens;
	std::string delimiters(",");
	StringUtils::Tokenize(str, tokens, delimiters);
	if ((int)tokens.size() != numChanStates)
	{
		std::cerr << "Expect a vector of 0s or 1s with "<< numChanStates << " elements" << std::endl;
	}
	for (int ii = 0; ii < numChanStates; ++ii )
	{
		vChannelStates[ii] = (int) atoi(tokens[ii].c_str());
	}

	// read in which state should be the initial state
	// ASSUMPTION: All channels having the same initial state
	jumpOverCommentLine(fpF);
  c = fgets(bufS, LENGTH_LINE_MAX, fpF);
  if (1 == sscanf(bufS, "%d ", &initialstate))
	{
		if (initialstate < 1 or initialstate > numChanStates)
		{
			std::cerr << "Syntax of Markov-model file invalid: expect \n index-of-initial-state" << std::endl;
			std::cerr << " to be in the range [1, " << numChanStates << "]\n";
			exit(EXIT_FAILURE);
		}
		// map to zero-based
		initialstate--;
	}
	else{
		std::cerr << "Please input only 1 value for channel's initial-state in Markov file\n";
		exit(EXIT_FAILURE);
	}
	

	// read in transition rate matrix
	matChannelRateConstant = new dyn_var_t*[numChanStates];
	for(int i = 0; i < numChanStates; ++i) {
		matChannelRateConstant[i] = new dyn_var_t[numChanStates]();
	}
	
	bool isOK = true;
	jumpOverCommentLine(fpF);
  c = fgets(bufS, LENGTH_LINE_MAX, fpF);
  while (!feof(fpF))
	{
		int ifrom, ito;
		float rate;
		if (3 != sscanf(bufS, "%d, %d, %f ", &ifrom, &ito, &rate))
		{
			isOK = false;
			break;
		}
		if (ifrom < 1 or ifrom > numChanStates or ito < 1 or ito > numChanStates )
		{
			isOK = false;
			break;
		}
		matChannelRateConstant[ifrom-1][ito-1] = rate;
		jumpOverCommentLine(fpF);
		c = fgets(bufS, LENGTH_LINE_MAX, fpF);
	}
	if (!isOK)
	{
		std::cerr << "Error at reading matrix of rate-constants" << std::endl;
		exit(EXIT_FAILURE);
	}
  fclose(fpF);
}

// NOTE
// matChannelRateConstant = [numChanStates][numChanStates] matrix of rate constant
// vChannelStates = [numChanStates] vector telling which state is conducting
// initialstate = value telling which state is the initial state for all channels in the cluster
void Params::readMarkovModel(const std::string& fname, dyn_var_t* &matChannelRateConstant,
		int &numChanStates, int* &vChannelStates, int &initialstate)
{
  FILE* fpF = fopen(fname.c_str(), "r");
  char bufS[LENGTH_LINE_MAX];
	if (fpF == NULL)
	{
		std::cerr << "File " << fname << " not found.\n";
		assert(fpF);
	}
	//read in the number of states
	jumpOverCommentLine(fpF);
  char* c = fgets(bufS, LENGTH_LINE_MAX, fpF);
  if (1 != sscanf(bufS, "%d ", &numChanStates))
	{
    std::cerr << "Syntax of Markov-model file invalid: expect \n #-states" << std::endl;
		exit(EXIT_FAILURE);
	}

	//read in which state(s) is open-state
	jumpOverCommentLine(fpF);
  vChannelStates =	new int[numChanStates]();
  c = fgets(bufS, LENGTH_LINE_MAX, fpF);
	std::string str(bufS);
	std::vector<std::string> tokens;
	std::string delimiters(",");
	StringUtils::Tokenize(str, tokens, delimiters);
	if ((int)tokens.size() != numChanStates)
	{
		std::cerr << "Expect a vector of 0s or 1s with "<< numChanStates << " elements" << std::endl;
	}
	for (int ii = 0; ii < numChanStates; ++ii )
	{
		vChannelStates[ii] = (int) atoi(tokens[ii].c_str());
	}

	// read in which state should be the initial state
	// ASSUMPTION: All channels having the same initial state
	jumpOverCommentLine(fpF);
  c = fgets(bufS, LENGTH_LINE_MAX, fpF);
  if (1 == sscanf(bufS, "%d ", &initialstate))
	{
		if (initialstate < 1 or initialstate > numChanStates)
		{
			std::cerr << "Syntax of Markov-model file invalid: expect \n index-of-initial-state" << std::endl;
			std::cerr << " to be in the range [1, " << numChanStates << "]\n";
			exit(EXIT_FAILURE);
		}
		// map to zero-based
		initialstate--;
	}
	else{
		std::cerr << "Please input only 1 value for channel's initial-state in Markov file\n";
		exit(EXIT_FAILURE);
	}
	

	// read in transition rate matrix
	matChannelRateConstant = new dyn_var_t[numChanStates* numChanStates]();
	
	bool isOK = true;
	jumpOverCommentLine(fpF);
  c = fgets(bufS, LENGTH_LINE_MAX, fpF);
  while (!feof(fpF))
	{
		int ifrom, ito;
		float rate;
		if (3 != sscanf(bufS, "%d, %d, %f ", &ifrom, &ito, &rate))
		{
			isOK = false;
			break;
		}
		if (ifrom < 1 or ifrom > numChanStates or ito < 1 or ito > numChanStates )
		{
			isOK = false;
			break;
		}
		matChannelRateConstant[Map1Dindex(ifrom-1,ito-1,numChanStates)] = rate;
		jumpOverCommentLine(fpF);
		c = fgets(bufS, LENGTH_LINE_MAX, fpF);
	}
	if (!isOK)
	{
		std::cerr << "Error at reading matrix of rate-constants" << std::endl;
		exit(EXIT_FAILURE);
	}
  fclose(fpF);
}

// Cluster:
// Suppose a cluster has M states
// M = numClusterStates  = number of states in the cluster
// the transition matrix is of size M * M
// which is sparse
// NOTE:
//
// numClusterStates = scalar
//
// matClusterStateInfo[M*numChanStates] = 2D matrix, each row
//                is a vector of length 'numChanStates' telling cluster-state info
//                which is how many channels in each channel-state
//
// vClusterNumOpenChan   = vector of size M
//          that tells the #openingChannel at that cluste-state
//
// maxNumNeighbors = max-number of neighbors for all rows
//
// matK_channelstate_fromto[0..numClusterStates-1][0..maxNumNeighbors-1] = {from|to}
//    'from' and 'to' are both index of single-channel state
// matK_indx  = the real index  of cluster-state
// matK_indx[i][j] = val=> keeping the real index of the next state
//               which can be used to trace 
//               to the next state using indexK[val][?]
void Params::setupCluster(
		dyn_var_t* const &matChannelRateConstant,
		const int &numChan, const int &numChanStates, int* const &vChannelStates, 
		//output
    int & numClusterStates, 
		int* &matClusterStateInfo, 
		int* &vClusterNumOpenChan,
		int &maxNumNeighbors,
		//Combined2MarkovState_t* &matK_channelstate_fromto, long * &matK_indx
		long* &matK_channelstate_fromto, ClusterStateIndex_t * &matK_indx
		)
{
	// Step 1. find numClusterStates, matClusterStateInfo
	int balls = numChan;
	int bins = numChanStates;
	// how many ways to put balls
	// into bins
	// regardless of which balls
    //CALL count_ball2bin(balls, bins, matClusterStateInfo, numClusterStates, icols)
		//matClusterStateInfo [0..ClusterNumStates-1,0..numChanStates-1]
		//each row of matClusterStateInfo represent each cluster-state, i.e. it tells
		//the information of Channels distribution to each Markov-state
	int rows, cols;
	count_ball2bin(balls, bins, matClusterStateInfo, rows, cols);
	numClusterStates = rows;
	assert(bins=cols);

	// Step 2. Find vClusterNumOpenChan
  // + vector that tells how many conducting channels in each cluster-state
	vClusterNumOpenChan = new int[numClusterStates]();

	for (unsigned int ii=0; ii < numClusterStates; ii++)
	{
		for (unsigned int jj=0; jj< numChanStates; jj++)
		{
			int OpenState = 1;
			if (vChannelStates[jj] == OpenState)
			{
				vClusterNumOpenChan[ii] += matClusterStateInfo[Map1Dindex(ii,jj, numChanStates)];
			}
		}
	}
	
	// Step 3. find maxNumNeighbors, matK_channelstate_fromto, matK_indx
/*
    !size (numClusterStates,0:maxnkL..) :: compK_L..
    !size (numClusterStates,0:maxnkL..) :: idxK_L..
    ! NOTE: The zero-th column keeps the 'true' number of non-zero elements in the row
    !       However, it's does't mean anything here as non-zero elements are not consecutives all the time
    CALL getcompK_5(channel_cMat, mL, matClusterStateInfo, numClusterStates, N_L, matK_channelstate_fromto, matK_indx, maxNumNeighbors)
*/

	getCompactK(matChannelRateConstant, 
			numChanStates,
			numChan, 
			matClusterStateInfo, 
			numClusterStates,
			//output
			maxNumNeighbors,
			matK_channelstate_fromto,
			matK_indx
			);
}

//NOTE:
//matK_channelstate_fromto[..][..] = numClusterStates * maxNumNeighbors
//matK_indx[..][..] = of the same size
void Params::getCompactK(
	dyn_var_t* const & matChannelRateConstant,
  const int & numChanStates,
  const int & numChan,
  int* const &matClusterStateInfo,
  const int & numClusterStates,
	 // output
  int & maxNumNeighbors,
  long* & matK_channelstate_fromto,
  ClusterStateIndex_t* & matK_indx
		)
{
	// Step 1. Find maximum # of possible neighbors
	// which is equal to # of non-zero non-diagonal elements
	unsigned int icount = 0;
	if (numChan > numChanStates)
	{
		for (int ii=0; ii< numChanStates; ii++)
		{
			int offset = Map1Dindex(ii,0, numChanStates);
			icount += count_nonzero(matChannelRateConstant, offset, numChanStates );
		}
	}
	else{
		//a more general case 
		//which requires sorting
		//and takes only the sum of the min(numChanStates,numChan) rows of the most non-zero elements
		std::vector<int> vcount;
		for (int ii = 0; ii< numChanStates; ii++)
		{
			int offset = Map1Dindex(ii,0, numChanStates);
			int icount = count_nonzero(matChannelRateConstant, offset, numChanStates );
			vcount.push_back(icount);
		}
		std::sort(vcount.begin(), vcount.end(), std::greater<int>());//descending
		icount = 0;
		for(int ii=0; ii < std::min(numChanStates, numChan); ii++)
			icount  += vcount[ii];
	}
	maxNumNeighbors = icount;

	// Step 2. Find big K matrice in compact form
  matK_channelstate_fromto = new long[numClusterStates * maxNumNeighbors]();
	//matK_indx = new int[numClusterStates * maxNumNeighbors]();
	matK_indx = new ClusterStateIndex_t[numClusterStates * maxNumNeighbors]();

	long fromto;
	for (int ii=0; ii < numClusterStates; ii++)
	{
		int icount = 0;
		//	std::valarray<int> v1(matClusterStateInfo + Map1Dindex(ii,0,numChanStates), numChanStates);
		std::vector<int> v1(matClusterStateInfo + Map1Dindex(ii,0, numChanStates), 
				matClusterStateInfo + Map1Dindex(ii,0, numChanStates) + numChanStates 
				);
/*		std::cout << "v1: ";
		for (int jj = 0; jj < v1.size(); jj++)
			std::cout << v1[jj] << ",";
		std::cout << std::endl;
		*/
		for (int jj=0; jj < numClusterStates; jj++)
		{
//			//std::valarray<int> v2(matClusterStateInfo + Map1Dindex(jj,0,numChanStates), numChanStates);
			std::vector<int> v2(matClusterStateInfo + Map1Dindex(jj,0,numChanStates), 
					matClusterStateInfo + Map1Dindex(jj,0,numChanStates) + numChanStates);
//			//std::valarray<int> res = (v2 - v1);
			std::vector<int> res;
			res.reserve(v1.size());
			//std::transform(v2.begin(), v2.end(), v1.begin(), res.begin(), std::minus<int>());
			std::transform(v2.begin(), v2.end(), v1.begin(), std::back_inserter(res), std::minus<int>());
			// int sumval = std::abs(res.sum());
			int sumval = std::accumulate(res.begin(), res.end(), 0, [](int a, int b){ return std::abs(a)+std::abs(b); });
/*		std::cout << "v2: ";
		for (int jj = 0; jj < v2.size(); jj++)
			std::cout << v2[jj] << ",";
		std::cout << std::endl;
		std::cout << "--- v2-v1: ";
		for (int jj = 0; jj < res.size(); jj++)
			std::cout << res[jj] << ",";
		std::cout << std::endl;
		std::cout << "sumval = " << sumval ;
		std::cin.get();
		*/
		if (sumval == 2) // eligible for two neighbor cluster-states
		{
			//				//int initial_state = std::distance(res, std::max_element(std::begin(res), res.size())); //location of element with value 1;
			//				//int final_state = std::distance(res, std::min_element(res, res.size())) //location of element with value -1;
			int initial_state = std::distance(res.begin(), std::max_element(res.begin(), res.end())); //location of element with value 1;
			int final_state = std::distance(res.begin(), std::min_element(res.begin(), res.end())); //location of element with value -1;
			if (initial_state > MAXRANGE_MARKOVSTATE or final_state > MAXRANGE_MARKOVSTATE)
			{
				std::cerr << "The Markov-model should not have more than " << MAXRANGE_MARKOVSTATE << " states\n";
				exit(EXIT_FAILURE);
			}
			//std::cout << "from, to = " << initial_state << ", " << final_state << std::endl;
			if (matChannelRateConstant[Map1Dindex(initial_state, final_state, numChanStates)] > 0.0)
			{
/*
               fromto = IOR(ISHFT(INT(initial_state), bitshift), final_state)
                ChannelStateFromTo(ii, icount) = fromto ![initial_state,final_state]
                idxK(ii, icount) = jj
*/
 
				fromto = final_state | (initial_state << BITSHIFT_MARKOV);
				matK_channelstate_fromto[Map1Dindex(ii, icount, maxNumNeighbors)] = fromto;
				matK_indx[Map1Dindex(ii, icount, maxNumNeighbors)] = jj;
				icount++;
			}
		}
		}
	}
}
