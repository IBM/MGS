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

#include <string.h>
#include <sstream>

#define LENGTH_LINE_MAX 1024
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
      _SIParams(false),
      _compartmentVariables(false),
      _channels(false),
      _electricalSynapses(false),
      _chemicalSynapses(false)
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

void Params::readDevParams(const std::string& fname)
{
  FILE* fpF = fopen(fname.c_str(), "r");
  assert(fpF);
  skipHeader(fpF);
  assert(readBondParams(fpF));
  // assert(readAngleParams(fpF));
  assert(readLJParams(fpF));
  readSIParams(fpF);
  readRadii(fpF);
  fclose(fpF);
}

void Params::readDetParams(const std::string& fname)
{
  FILE* fpF = fopen(fname.c_str(), "r");
  assert(fpF);
  skipHeader(fpF);
  readRadii(fpF);
  readTouchTables(fpF);
  fclose(fpF);
}

void Params::readCptParams(const std::string& fname)
{
  FILE* fpF = fopen(fname.c_str(), "r");
  assert(fpF);
  skipHeader(fpF);
  readCompartmentVariableTargets(fpF);
  readCompartmentVariableCosts(fpF);
  readModelParams(fpF, "COMPARTMENT_VARIABLE_PARAMS", _compartmentParamsMasks,
                  _compartmentParamsMap, _compartmentArrayParamsMap);
  fclose(fpF);
}

void Params::readChanParams(const std::string& fname)
{
  FILE* fpF = fopen(fname.c_str(), "r");
  assert(fpF);
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
  assert(fpF);
  skipHeader(fpF);

  std::string keyword;
  keyword = std::string("ELECTRICAL_SYNAPSE_TARGETS");
  if (isGivenKeywordNext(fpF, keyword))
  {
    readElectricalSynapseTargets(fpF);
    readElectricalSynapseCosts(fpF);
  }
  keyword = std::string("BIDIRECTIONAL_CONNECTION_TARGETS");
  if (isGivenKeywordNext(fpF, keyword))
  {
    readBidirectionalConnectionTargets(fpF);
    readBidirectionalConnectionCosts(fpF);
  }
  keyword = std::string("CHEMICAL_SYNAPSE_TARGETS");
  if (isGivenKeywordNext(fpF, keyword))
  {
    readChemicalSynapseTargets(fpF);
    readChemicalSynapseCosts(fpF);
  }
  // readElectricalSynapseTargets(fpF);
  // readElectricalSynapseCosts(fpF);
  // readBidirectionalConnectionTargets(fpF);
  // readBidirectionalConnectionCosts(fpF);
  // readChemicalSynapseTargets(fpF);
  // readChemicalSynapseCosts(fpF);
  readPreSynapticPointTargets(fpF);
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

std::list<Params::Params::ElectricalSynapseTarget>*
    Params::getElectricalSynapseTargets(key_size_t key1, key_size_t key2)
{
  std::list<Params::Params::ElectricalSynapseTarget>* rval = 0;
  if (_electricalSynapses)
  {  // if there is information about what branch can connect with what branch
    // to form a bidirectional connection
    // then
    std::map<key_size_t,
             std::map<key_size_t,
                      std::list<Params::Params::ElectricalSynapseTarget> > >::
        iterator miter1 =
            _electricalSynapseTargetsMap.find(_segmentDescriptor.getSegmentKey(
                key1, _electricalSynapseTargetsMask1));
    if (miter1 != _electricalSynapseTargetsMap.end())
    {
      std::map<key_size_t,
               std::list<Params::Params::ElectricalSynapseTarget> >::iterator
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

std::list<Params::Params::BidirectionalConnectionTarget>*
    Params::getBidirectionalConnectionTargets(key_size_t key1, key_size_t key2)
{
  std::list<Params::Params::BidirectionalConnectionTarget>* rval = 0;
  if (_bidirectionalConnections)
  {  // if there is information about what branch can connect with what branch
    // to form a bidirectional connection
    // then
    std::map<
        key_size_t,
        std::map<key_size_t,
                 std::list<Params::Params::BidirectionalConnectionTarget> > >::
        iterator miter1 = _bidirectionalConnectionTargetsMap.find(
            _segmentDescriptor.getSegmentKey(
                key1, _bidirectionalConnectionTargetsMask1));
    if (miter1 != _bidirectionalConnectionTargetsMap.end())
    {
      std::map<
          key_size_t,
          std::list<Params::Params::BidirectionalConnectionTarget> >::iterator
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
    exit(0);
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
    exit(0);
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
    exit(0);
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
    exit(0);
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
    exit(0);
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

// A comment line is a blank line
//  or               a line whose first non-space character is #
bool Params::isCommentLine(std::string& line)
{
  bool rval = false;
  char space = ' ';
  int index = line.find_first_not_of(space);
  if (index != std::string::npos)
  {
    if (line[index] == '#' or line[index] == '\n') rval = true;
  }
  else if (line.length() == 0)
    rval = true;
  return rval;
}

// Skip the comment lines
void Params::jumpOverCommentLine(FILE* fpF)
{
  fpos_t fpos;
  fgetpos(fpF, &fpos);
  char bufS[LENGTH_LINE_MAX], tokS[LENGTH_TOKEN_MAX];
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
        std::cerr << "Expected: " << keyword << ", but Found: " << btype << std::endl;
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
      for (int j = 0; j < sz;
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
    unsigned int sz = maskVector.size() * 2;
    unsigned int* ids = new unsigned int[sz];

    for (int i = 0; i < n;i++)  // for each non-comment line
    {
      jumpOverCommentLine(fpF);
      for (int j = 0; j < sz; ++j)
      {
        if (1 != fscanf(fpF, "%d", &ids[j])) assert(0);
        if (maskVector[j] ==
            SegmentDescriptor::branchType)  // special treatment
          // for NSITYPES, BRANCHTYPE is used twice
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
    unsigned int* ids = new unsigned int[sz];

    for (int i = 0; i < n; i++)  // for each line, not counting comment-line
    {
      jumpOverCommentLine(fpF);
      for (int j = 0; j < sz; ++j)
      {
        if (1 != fscanf(fpF, "%d", &ids[j])) assert(0);
        if (maskVector[j] == SegmentDescriptor::branchType)
          ids[j] = ids[j] - 1;  // make sure the BRANCHTYPE is 0-based
        if (maskVector[j] == SegmentDescriptor::segmentIndex)
        {
          std::cerr << "Params : Targeting compartmentVariables to individual "
                       "compartments not supported!" << std::endl;
          exit(0);
        }
      }
      assert(!feof(fpF));
      c = fgets(bufS, LENGTH_LINE_MAX, fpF);
      std::istringstream is(bufS);
      std::list<std::string> targets;
      std::string type;
      while (is >> type)
      {
        targets.push_back(type);
      }
      targets.sort();
      _compartmentVariableTargetsMap[_segmentDescriptor.getSegmentKey(
          maskVector, &ids[0])] = targets;
    }
    delete[] ids;
  }
  else
    rval = false;
  _compartmentVariables = rval;
  return _compartmentVariables;
}

bool Params::readChannelTargets(FILE* fpF)
{
  /*
CHANNEL_TARGETS 8
BRANCHTYPE MTYPE
1 0 HCN [Voltage] [Voltage] Nat [Voltage] [Voltage]
*/
  _channels = false;
  bool rval = true;
  _channelTargetsMask = 0;
  _channelTargetsMap.clear();
  skipHeader(fpF);
  int n = 0;
  char bufS[LENGTH_LINE_MAX], tokS[LENGTH_TOKEN_MAX];
  jumpOverCommentLine(fpF);
  char* c = fgets(bufS, LENGTH_LINE_MAX, fpF);
  if (2 == sscanf(bufS, "%s %d ", tokS, &n))
  {
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
    unsigned int* ids = new unsigned int[sz];

    for (int i = 0; i < n; i++)  // for each line, not counting comment-line
    {
      jumpOverCommentLine(fpF);
      for (unsigned int j = 0; j < sz; ++j)
      {
        if (1 != fscanf(fpF, "%d", &ids[j])) assert(0);
        if (maskVector[j] == SegmentDescriptor::branchType)
          ids[j] = ids[j] - 1;  // make sure the BRANCHTYPE is 0-based
        if (maskVector[j] == SegmentDescriptor::segmentIndex)
        {
          std::cerr << "Params : Targeting channels to individual compartments "
                       "not supported!" << std::endl;
          exit(0);
        }
      }
      assert(!feof(fpF));
      c = fgets(bufS, LENGTH_LINE_MAX, fpF);
      std::istringstream is(bufS);
      std::list<Params::ChannelTarget>& targets =
          _channelTargetsMap[_segmentDescriptor.getSegmentKey(maskVector,
                                                              &ids[0])];
      Params::ChannelTarget ct;
      while (is >> ct._type)
      {
        while (is.get() != '[')
        {
          assert(is.good());
        }
        char buf1[LENGTH_IDNAME_MAX];
        is.get(buf1, LENGTH_IDNAME_MAX, ']');

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
        is.get(buf2, LENGTH_IDNAME_MAX, ']');
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
    delete[] ids;
  }
  else
    rval = false;

  _channels = rval;
  return _channels;
}

bool Params::readElectricalSynapseTargets(FILE* fpF)
{
  _electricalSynapses = false;
	bool rval = true;
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
	else rval = false;

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

    for (int i = 0; i < n;i++)  // for each line, not counting comment-line
		{
			jumpOverCommentLine(fpF);
			// one line:
			// 2 2     2 0   DenSpine [Voltage] 1.0
			for (int j = 0; j < sz1; ++j)
			{
				if (1 != fscanf(fpF, "%d", &ids1[j])) assert(0);
				if (maskVector1[j] == SegmentDescriptor::branchType)
					ids1[j] = ids1[j] - 1;  // make sure the BRANCHTYPE is 0-based
			}                           // read-in 2 2
			for (int j = 0; j < sz2; ++j)
			{
				if (1 != fscanf(fpF, "%d", &ids2[j])) assert(0);
				if (maskVector2[j] == SegmentDescriptor::branchType)
					ids2[j] = ids2[j] - 1;  // make sure the BRANCHTYPE is 0-based
			}                           // read-in 2 0
			assert(!feof(fpF));
			c = fgets(bufS, LENGTH_LINE_MAX,
					fpF);  // read-in DenSpine [Voltage] 1.0
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
				is.get(buf, LENGTH_IDNAME_MAX, ']');
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
	else rval = false;
	_electricalSynapses = rval;
  return _electricalSynapses;
}

bool Params::readBidirectionalConnectionTargets(FILE* fpF)
{
	bool rval = true;
  _bidirectionalConnections = false;
  _bidirectionalConnectionTargetsMask1 = _bidirectionalConnectionTargetsMask2 =
      0;
  _bidirectionalConnectionTargetsMap.clear();
  skipHeader(fpF);
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
	}else rval = false;

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

		for (int i = 0; i < n;i++)  // for each line, not counting comment-line
		{
			jumpOverCommentLine(fpF);
			// one line:
			// 2 2     2 0   DenSpine [Voltage, Calcium] 1.0
			for (int j = 0; j < sz1; ++j)
			{
				if (1 != fscanf(fpF, "%d", &ids1[j])) assert(0);
				if (maskVector1[j] == SegmentDescriptor::branchType)
					ids1[j] = ids1[j] - 1;  // make sure the BRANCHTYPE is 0-based
			}                           // read-in 2 2
			for (int j = 0; j < sz2; ++j)
			{
				if (1 != fscanf(fpF, "%d", &ids2[j])) assert(0);
				if (maskVector2[j] == SegmentDescriptor::branchType)
					ids2[j] = ids2[j] - 1;  // make sure the BRANCHTYPE is 0-based
			}                           // read-in 2 0
			assert(!feof(fpF));
			c = fgets(bufS, LENGTH_LINE_MAX,
					fpF);  // read-in DenSpine [Voltage] 1.0
			std::istringstream is(bufS);

			std::map<key_size_t, std::list<Params::BidirectionalConnectionTarget> >&
				targetsMap = _bidirectionalConnectionTargetsMap
				[_segmentDescriptor.getSegmentKey(maskVector1, &ids1[0])];
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
				is.get(buf, LENGTH_IDNAME_MAX, ']');
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
	else rval = false;
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
	else rval = false;

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

    for (int i = 0; i < n;i++)
    {
        for (int j = 0; j < sz1; ++j)
        {
          if (1 != fscanf(fpF, "%d", &ids1[j])) assert(0);
          if (maskVector1[j] == SegmentDescriptor::branchType)
            ids1[j] = ids1[j] - 1;  // make sure the BRANCHTYPE is 0-based
        }
        for (int j = 0; j < sz2; ++j)
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
        is.get(buf1, LENGTH_IDNAME_MAX, ']');
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
          is.get(buf1, LENGTH_IDNAME_MAX, ']');
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
          is.get(buf2, LENGTH_IDNAME_MAX, ']');
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
	else rval = false;
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
  bool rval = false;
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
					//do nothing
				}else rval = false;
      }else rval = false;
  if (n > 0)
  {
    for (int i = 0; i < n;i++)  // for each line, not counting comment-line
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
  }else rval = false;
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
			//do nothing
		}else rval = false;
	}else rval = false;

  assert(n > 0);

  std::vector<SegmentDescriptor::SegmentKeyData> maskVector;
  mask = resetMask(fpF, maskVector);
  double p;
  unsigned int sz = maskVector.size();
  assert(sz);
  unsigned int* ids = new unsigned int[sz];

  for (int i = 0; i < n;i++)  // for each line, not counting comment-line
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
  /* NOTE: id = COMPARTMENT_VARIABLE_PARAMS
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
      /* One group is
Calcium 3
MTYPE BRANCHTYPE
0 1 <CaClearance=1.1>
0 3 <CaClearance=4.2>
0 4 <CaClearance=4.2>
       */
      jumpOverCommentLine(fpF);
      int p;
      c = fgets(bufS, LENGTH_LINE_MAX, fpF);
      if (2 == sscanf(bufS, "%s %d ", tokS, &p))// e.g.: Calcium 3
      {  
        std::string modelID(tokS);
        std::vector<SegmentDescriptor::SegmentKeyData> maskVector;
        paramsMasks[modelID] = resetMask(fpF, maskVector);
        unsigned int sz = maskVector.size();
        assert(sz);
        unsigned int* ids = new unsigned int[sz];
        for (int j = 0; j < p;
             j++)  // for each line (subgroup), not counting comment-line
        {
          jumpOverCommentLine(fpF);
          for (int k = 0; k < sz; ++k)
          {  // read vector mask part
            if (1 != fscanf(fpF, "%d", &ids[k])) assert(0);
            if (maskVector[k] == SegmentDescriptor::branchType)
              ids[k] = ids[k] - 1;  // make sure the BRANCHTYPE is 0-based
            if (maskVector[k] == SegmentDescriptor::segmentIndex)
            {
              std::cerr << "Params : Targeting channel parameters to "
                           "individual compartments not supported!"
                        << std::endl;
              exit(0);
            }
          }
          assert(!feof(fpF));
          c = fgets(bufS, LENGTH_LINE_MAX, fpF);
          std::istringstream is(bufS);
          std::list<std::pair<std::string, dyn_var_t> >& params =
              paramsMap[modelID][_segmentDescriptor.getSegmentKey(maskVector,
                                                                  &ids[0])];
          std::list<std::pair<std::string, std::vector<dyn_var_t> > >&
              arrayParams =
                  arrayParamsMap[modelID][_segmentDescriptor.getSegmentKey(
                      maskVector, &ids[0])];
          while (is.get() != '<')
          {
            assert(is.good());
          }
          char buf1[LENGTH_IDNAME_MAX];
          is.get(buf1, LENGTH_IDNAME_MAX, '>');

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
            (*ii).erase(0, pos + delimiter.length());

            delimiter = " =";
            pos = (*ii).find(delimiter);
            std::string tok2 = (*ii).substr(0, pos);

            std::istringstream is2(tok2);
            if (is2.get() != '{')
            {
              dyn_var_t value = atof(tok2.c_str());
              params.push_back(std::pair<std::string, dyn_var_t>(name, value));
            }
            else
            {
              std::vector<dyn_var_t> value;
              // TUAN: potential bug if it takes more than 256 chars to see
              // '}'
              // need to be fixed soon
              char buf2[LENGTH_IDNAME_MAX];
              is2.get(buf2, LENGTH_IDNAME_MAX, '}');
              std::string stringbuf(buf2);
              std::vector<std::string> tokens;
              StringUtils::Tokenize(stringbuf, tokens, ",");
              for (std::vector<std::string>::iterator jj = tokens.begin(),
                                                      end = tokens.end();
                   jj != end; ++jj)
              {
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
        delete[] ids;
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
