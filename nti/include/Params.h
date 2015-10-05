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

#ifndef PARAMS_H
#define PARAMS_H

#include <cassert>
#include <math.h>
#include <vector>
#include <list>
#include <map>
//#include <string.h>
#include <string>
#include <algorithm>
#include <utility>

#include "SegmentDescriptor.h"

class Segment;

class SIParameters{
 public:
  SIParameters() : Epsilon(0.0), Sigma(1.0) {}
  double Epsilon;
  double Sigma;
};

class Params
{
public:
  class ChannelTarget
  {
   public: 
    ChannelTarget() : _type("") {}
    ChannelTarget(const ChannelTarget& ct) 
      : _type(ct._type), _target1(ct._target1), 
        _target2(ct._target2) {}
    std::string  _type;
    std::list<std::string> _target1, _target2;
    void addTarget1(std::string t1) {
      _target1.push_back(t1);
      _target1.sort();
    }
    void addTarget2(std::string t2) {
      _target2.push_back(t2);
      _target2.sort();
    }
    void clear() {
      _type="";
      _target1.clear();
      _target2.clear();
    }
    bool operator<(const ChannelTarget& ct)
    {
      bool rval=(_type<ct._type);
      if (_type==ct._type) {
	rval=_target1<ct._target1;
	if (_target1==ct._target1) {
	  rval=_target2<ct._target2;
	}
      }
      return rval;
    }
  };

  class ElectricalSynapseTarget
  {
   public: 
    ElectricalSynapseTarget() : _type(""), _parameter(0) {}
    ElectricalSynapseTarget(const ElectricalSynapseTarget& st) 
      : _type(st._type), _target(st._target), _parameter(st._parameter) {}
    std::string  _type;
    std::list<std::string> _target;
    double _parameter;
    void addTarget(std::string target) {
      _target.push_back(target);
      _target.sort();
    }
    void clear() {
      _type="";
      _target.clear();
      _parameter=0;
    }
    bool operator<(const ElectricalSynapseTarget& st)
    {
      bool rval=(_type<st._type);
      if (_type==st._type) {
	rval=_target<st._target;
      }
      return rval;
    }
  };
  
  class ChemicalSynapseTarget
  {
   public: 
    ChemicalSynapseTarget() : _parameter(0) {}
    ChemicalSynapseTarget(const ChemicalSynapseTarget& st) 
      : _targets(st._targets), _parameter(st._parameter) {}
     std::map<std::string, std::pair<std::list<std::string>, std::list<std::string> > > _targets;
     double _parameter;
     void addTarget1(std::string type, std::string target1) {
       _targets[type].first.push_back(target1);
     }
     void addTarget2(std::string type, std::string target2) {
       _targets[type].second.push_back(target2);
     }
     void clear() {
       _targets.clear();
       _parameter=0;
     }
     bool operator<(const ChemicalSynapseTarget& st)
     {
       bool rval=_targets<st._targets;
       return rval;
     }
  };

  Params();
  Params(Params const &);
  Params(Params&);
  ~Params();

  typedef enum{COMPARTMENT, CHANNEL, SYNAPSE} ModelType;

  void readDevParams(const std::string& fname);
  void readDetParams(const std::string& fname);
  void readCptParams(const std::string& fname);
  void readChanParams(const std::string& fname);
  void readSynParams(const std::string& fname);

  bool SIParams() {return _SIParams;}
  bool compartmentVariables() {return _compartmentVariables;}
  bool channels() {return _channels;}
  bool electricalSynapses() {return _electricalSynapses;}
  bool chemicalSynapses() {return _chemicalSynapses;}
  bool symmetricElectricalSynapseTargets(double key1, double key2);

  double getBondK0(int typ){assert(typ<_nBondTypes); return _bondK0[typ];}
  double getBondR0(int typ){assert(typ<_nBondTypes); return _bondR0[typ];}
  double getAngleK0(int typ){assert(typ<_nAngleTypes); return _angleK0[typ];}
  double getAngleR0(int typ){assert(typ<_nAngleTypes); return _angleR0[typ];}
  double getLJEps(int typ){assert(typ<_nLJTypes); return _ljEps[typ];}
  double getLJSigma(int typ){assert(typ<_nLJTypes); return _ljR0[typ];}

  double getRadius(double key);
  SIParameters getSIParams(double key1, double key2);
  std::list<std::string> const * getCompartmentVariableTargets(double key);
  std::list<ChannelTarget> * getChannelTargets(double key);
  std::list<ElectricalSynapseTarget> * getElectricalSynapseTargets(double key1, double key2);
  std::list<ChemicalSynapseTarget> * getChemicalSynapseTargets(double key1, double key2);
  std::string getPreSynapticPointTarget(std::string chemicalSynapseType);
  std::list<std::string>& getPreSynapticPointSynapseTypes(std::string targetType);

  bool isCompartmentVariableTarget(double key, std::string type);
  bool isChannelTarget(double key);
  bool isElectricalSynapseTarget(double key1, double key2, bool autapses);
  bool isElectricalSynapseTarget(double key);
  bool isChemicalSynapseTarget(double key1, double key2, bool autapses);
  bool isChemicalSynapseTarget(double key);

  double getCompartmentVariableCost(std::string compartmentVariableId);
  double getChannelCost(std::string channelId);
  double getElectricalSynapseCost(std::string electricalSynapseId);
  double getChemicalSynapseCost(std::string chemicalSynapseId);

  void getModelParams(ModelType modelType, std::string nodeType, double key, std::list<std::pair<std::string, float> >& modelParams);
  void getModelArrayParams(ModelType modelType, std::string nodeType, double key, std::list<std::pair<std::string, std::vector<float> > >& modelArrayParams);
  void getTouchTableMasks(std::vector<std::vector<SegmentDescriptor::SegmentKeyData> >& masks) {masks=_touchTableMasks;}

  void skipHeader(FILE* fpF);

  Params& operator=(const Params &p) {assert(0); return (*this);}

 private:
  bool readBondParams(FILE *fpF);
  bool readAngleParams(FILE *fpF);
  bool readLJParams(FILE *fpF);
  bool readRadii(FILE *fpF);
  bool readTouchTables(FILE *fpF);
  bool readSIParams(FILE *fpF);
  bool readCompartmentVariableTargets(FILE* fpF);
  bool readBranchPointTargets(FILE* fpF);
  bool readChannelTargets(FILE* fpF);
  bool readElectricalSynapseTargets(FILE* fpF);
  bool readChemicalSynapseTargets(FILE* fpF);
  bool readPreSynapticPointTargets(FILE* fpF);

  unsigned long long readNamedParam(FILE *fpF, std::string name, std::map<double, double>& namedParamsMap) ;

  bool readCompartmentVariableCosts(FILE *fpF);
  bool readChannelCosts(FILE *fpF);
  bool readElectricalSynapseCosts(FILE *fpF);
  bool readChemicalSynapseCosts(FILE *fpF);
  //bool readChannelParams(FILE* fpF);

  bool readModelParams(FILE* fpF, const std::string& id,
		       std::map<std::string, unsigned long long>& masks,
		       std::map<std::string, std::map<double, std::list<std::pair<std::string, float> > > >& paramsMap,
		       std::map<std::string, std::map<double, std::list<std::pair<std::string, std::vector<float> > > > >& paramsArrayMap);

  unsigned long long resetMask(FILE* fpF, std::vector<SegmentDescriptor::SegmentKeyData>& maskVector);

  double *_bondK0;
  double *_bondR0;
  int _nBondTypes;

  double *_angleK0;
  double *_angleR0;
  int _nAngleTypes;

  double *_ljEps;
  double *_ljR0;
  int _nLJTypes;

  unsigned long long _radiiMask, 
    _SIParamsMask, 
    _compartmentVariableTargetsMask,
    _channelTargetsMask, 
    _electricalSynapseTargetsMask1, 
    _electricalSynapseTargetsMask2, 
    _chemicalSynapseTargetsMask1, 
    _chemicalSynapseTargetsMask2;


  std::map<double, double> _radiiMap;
  std::map<double, std::map<double, SIParameters> > _SIParamsMap;
  std::map<double, std::list<std::string> > _compartmentVariableTargetsMap; 
  std::map<double, std::list<ChannelTarget> > _channelTargetsMap;
  std::map<std::string, unsigned long long> _channelParamsMasks;
  std::map<std::string, std::map<double, std::list<std::pair<std::string, float> > > > _channelParamsMap;
  std::map<std::string, std::map<double, std::list<std::pair<std::string, std::vector<float> > > > > _channelArrayParamsMap;
  std::map<std::string, unsigned long long> _compartmentParamsMasks;
  std::map<std::string, std::map<double, std::list<std::pair<std::string, float> > > > _compartmentParamsMap;
  std::map<std::string, std::map<double, std::list<std::pair<std::string, std::vector<float> > > > > _compartmentArrayParamsMap;
  std::map<double, std::map<double, std::list<ElectricalSynapseTarget> > > _electricalSynapseTargetsMap;
  std::map<double, std::map<double, std::list<ChemicalSynapseTarget> > >  _chemicalSynapseTargetsMap;
  std::map<std::string, std::string> _preSynapticPointTargetsMap;
  std::map<std::string, std::list<std::string> > _preSynapticPointSynapseMap;
  std::map<std::string, double> _compartmentVariableCostsMap,
    _channelCostsMap,
    _electricalSynapseCostsMap,
    _chemicalSynapseCostsMap;
  
  std::vector<std::vector<SegmentDescriptor::SegmentKeyData> > _touchTableMasks;

  bool _SIParams, _compartmentVariables, _channels, _electricalSynapses, _chemicalSynapses;

  SegmentDescriptor _segmentDescriptor;
};
#endif
