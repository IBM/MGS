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

#include "MaxComputeOrder.h"

#include <cassert>
#include <math.h>
#include <vector>
#include <list>
#include <map>
//#include <string.h>
#include <string>
#include <algorithm>
#include <utility>
#include <cstdint>
#include <climits>

#include "SegmentDescriptor.h"

#define Combined2MarkovState_t long

class Segment;

class SIParameters
{
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
        : _type(ct._type), _target1(ct._target1), _target2(ct._target2)
    {
    }
    std::string _type; //nodekind's value passed to GSL
    std::list<std::string> _target1, _target2; 
		  //_target1 = list of nodekind's value as input 
			//_target2 = list of nodekind's value as output
    void addTarget1(std::string t1)
    {
      _target1.push_back(t1);
      _target1.sort();
    }
    void addTarget2(std::string t2)
    {
      _target2.push_back(t2);
      _target2.sort();
    }
    void clear()
    {
      _type = "";
      _target1.clear();
      _target2.clear();
    }
    bool operator<(const ChannelTarget& ct)
    {
      bool rval = (_type < ct._type);
      if (_type == ct._type)
      {
        rval = _target1 < ct._target1;
        if (_target1 == ct._target1)
        {
          rval = _target2 < ct._target2;
        }
      }
      return rval;
    }
  };

  class ElectricalSynapseTarget
  {//i.e. gap junction
public:
    ElectricalSynapseTarget() : _type(""), _parameter(0) {}
    ElectricalSynapseTarget(const ElectricalSynapseTarget& st)
        : _type(st._type), _target(st._target), _parameter(st._parameter)
    {
    }
    std::string _type; //nodekind's value passed to GSL
    std::list<std::string> _target;
		  //_target = list of nodekind's value as passing in 2 directions 
			//  NOTE: designed for gap-junction, i.e.value should refer to voltage
    double _parameter;//the probability for forming the gap-junction 
    void addTarget(std::string target)
    {
      _target.push_back(target);
      _target.sort();
    }
    void clear()
    {
      _type = "";
      _target.clear();
      _parameter = 0;
    }
    bool operator<(const ElectricalSynapseTarget& st)
    {
      bool rval = (_type < st._type);
      if (_type == st._type)
      {
        rval = _target < st._target;
      }
      return rval;
    }
  };

	class BidirectionalConnectionTarget: public ElectricalSynapseTarget
	{//i.e. spine-attachment to shaft 
		  //_target = list of nodekind's value as passing in 2 directions 
			//  NOTE: designed for spine-attachment, i.e.value should refer to 
			//          electrical (i.e. voltage) and/or 
			//          chemical (e.g. calcium-cyto, calcium-ER, mobile-species)
			//          parts
	};

  class ChemicalSynapseTarget
  {
public:
    ChemicalSynapseTarget() : _parameter(0) {}
    ChemicalSynapseTarget(const ChemicalSynapseTarget& st)
        : _targets(st._targets), _parameter(st._parameter)
    {
    }
		  //_target = map a given synapse's receptor name
			//          with a pair of 'inputs', 'outputs'
			//   'inputs' = list of nodekind's value as input to the receptor above 
			//   'outputs' = list of nodekind's value as output of the receptor above
      //e.g. map<"AMPAR", pair<list{Voltage, Calcium}, list{Voltage}>
    std::map<std::string, std::pair<std::list<std::string>,
                                    std::list<std::string> > > _targets;
    double _parameter;//the probability for forming the synapse
    void addTarget1(std::string type, std::string target1)
    {
      _targets[type].first.push_back(target1);
    }
    void addTarget2(std::string type, std::string target2)
    {
      _targets[type].second.push_back(target2);
    }
    void clear()
    {
      _targets.clear();
      _parameter = 0;
    }
    bool operator<(const ChemicalSynapseTarget& st)
    {
      bool rval = _targets < st._targets;
      return rval;
    }
  };

  Params();
  Params(Params const&);
  Params(Params&);
  ~Params();

  typedef enum
  {
    COMPARTMENT,
    CHANNEL,
    SYNAPSE
  } ModelType;

  void readDevParams(const std::string& fname);
  void readDetParams(const std::string& fname);
  void readCptParams(const std::string& fname);
  void readChanParams(const std::string& fname);
  void readSynParams(const std::string& fname);

  bool SIParams() { return _SIParams; }
  bool compartmentVariables() { return _compartmentVariables; }
  bool channels() { return _channels; }
  bool electricalSynapses() { return _electricalSynapses; }
  bool bidirectionalConnections() { return _bidirectionalConnections; }
  bool chemicalSynapses() { return _chemicalSynapses; }
  bool symmetricElectricalSynapseTargets(key_size_t key1, key_size_t key2);
  bool symmetricBidirectionalConnectionTargets(key_size_t key1, key_size_t key2);

  double getBondK0(int typ)
  {
    assert(typ < _nBondTypes);
    return _bondK0[typ];
  }
  double getBondR0(int typ)
  {
    assert(typ < _nBondTypes);
    return _bondR0[typ];
  }
  double getAngleK0(int typ)
  {
    assert(typ < _nAngleTypes);
    return _angleK0[typ];
  }
  double getAngleR0(int typ)
  {
    assert(typ < _nAngleTypes);
    return _angleR0[typ];
  }
  double getLJEps(int typ)
  {
    assert(typ < _nLJTypes);
    return _ljEps[typ];
  }
  double getLJSigma(int typ)
  {
    assert(typ < _nLJTypes);
    return _ljR0[typ];
  }

  double getRadius(key_size_t key);
  SIParameters getSIParams(key_size_t key1, key_size_t key2);
  std::list<std::string> const* getCompartmentVariableTargets(key_size_t key);
  std::list<ChannelTarget>* getChannelTargets(key_size_t key);
  std::list<ElectricalSynapseTarget>* getElectricalSynapseTargets(key_size_t key1,
                                                                  key_size_t key2);
  std::list<BidirectionalConnectionTarget>* getBidirectionalConnectionTargets(key_size_t key1,
                                                                  key_size_t key2);
  std::list<ChemicalSynapseTarget>* getChemicalSynapseTargets(key_size_t key1,
                                                              key_size_t key2);
  std::string getPreSynapticPointTarget(std::string chemicalSynapseType);
  std::list<std::string>& getPreSynapticPointSynapseTypes(
      std::string targetType);

  bool isCompartmentVariableTarget(key_size_t key, std::string type);
  bool isChannelTarget(key_size_t key);
  bool isElectricalSynapseTarget(key_size_t key1, key_size_t key2, bool autapses);
  bool isElectricalSynapseTarget(key_size_t key);
	bool isGapJunctionTarget(key_size_t key);
	bool isGapJunctionTarget(key_size_t key1, key_size_t key2, bool autapses);
  bool isBidirectionalConnectionTarget(key_size_t key1, key_size_t key2, bool autapses);
  bool isBidirectionalConnectionTarget(key_size_t key);
  bool isChemicalSynapseTarget(key_size_t key1, key_size_t key2, bool autapses);
  bool isChemicalSynapseTarget(key_size_t key);

  double getCompartmentVariableCost(std::string compartmentVariableId);
  double getChannelCost(std::string channelId);
  double getElectricalSynapseCost(std::string electricalSynapseId);
  double getBidirectionalConnectionCost(std::string bidirectionalConnectionId);
  double getChemicalSynapseCost(std::string chemicalSynapseId);

  void getModelParams(ModelType modelType, std::string nodeType, key_size_t key,
                      std::list<std::pair<std::string, dyn_var_t> >& modelParams);
  void getModelParams(ModelType modelType, std::string nodeType, key_size_t key,
                      std::list<std::pair<std::string, std::string> >& modelParams);
  void getModelArrayParams(
      ModelType modelType, std::string nodeType, key_size_t key,
      std::list<std::pair<std::string, std::vector<dyn_var_t> > >&
          modelArrayParams);
  void getTouchTableMasks(
      std::vector<std::vector<SegmentDescriptor::SegmentKeyData> >& masks)
  {
    masks = _touchTableMasks;
  }

	void readMarkovModel(const std::string& fname, dyn_var_t** &matChannelRateConstant,
			int &numChanStates, int* &vChannelStates, int &initialstate);
	void readMarkovModel(const std::string& fname, dyn_var_t* &matChannelRateConstant,
			int &numChanStates, int* &vChannelStates, int &initialstate);
	void setupCluster(
			dyn_var_t* const &matChannelRateConstant,
			const int &numChan, 
			const int &numChanStates, int* const &vChannelStates, 
			//output
			int & numClusterStates, 
			int* &matClusterStateInfo, 
			int* &vClusterNumOpenChan,
			int &maxNumNeighbors,
			//Combined2MarkovState_t * &matK_channelstate_fromto, long* &indxK
			long* &matK_channelstate_fromto, ClusterStateIndex_t* &indxK
			);
	void getCompactK(
			dyn_var_t* const & matChannelRateConstant,
			const int & numChanStates,
			const int & numChan,
			int* const &matClusterStateInfo,
			const int & numClusterStates,
			// output
			int & maxNumNeighbors	,
			long* & matK_channelstate_fromto,
			//int* & matK_indx
			ClusterStateIndex_t* & matK_indx
			);

	Params& operator=(const Params& p)
	{
    assert(0);
    return (*this);
  }

  void skipHeader(FILE* fpF);
	void getListofValues(FILE* fpF, std::vector<int>& values);
  bool isCommentLine(std::string& line);
  void jumpOverCommentLine(FILE* fpF);

  bool isGivenKeySpineNeck(key_size_t key) ;
  bool isGivenKeySpineHead(key_size_t key) ;

  private:
	bool isGivenKeywordNext(FILE* fpF, std::string& keyword);
	std::string findNextKeyword(FILE* fpF);
	void readMultiLine(std::string& out_bufS, FILE* fpF);
	void buildChannelTargetsMap(
			std::vector<SegmentDescriptor::SegmentKeyData>& maskVector,
			std::vector<unsigned int*>& v_ids, 
			//std::istringstream &is
			const std::string & myBuf
			);
	void buildCompartmentVariableTargetsMap(
			std::vector<SegmentDescriptor::SegmentKeyData>& maskVector,
			std::vector<unsigned int*>& v_ids, 
			//std::istringstream& is
			const std::string & myBuf
			);
	void buildParamsMap(
			std::vector<SegmentDescriptor::SegmentKeyData>& maskVector,
			std::vector<unsigned int*>& v_ids, 
			//std::istringstream& is, 
			const std::string & myBuf,
			std::string & modelID,
			std::map<std::string,
			std::map<key_size_t, std::list<std::pair<std::string, dyn_var_t> > > >&
			paramsMap,
			std::map<
			std::string,
			std::map<key_size_t,
			std::list<std::pair<std::string, std::vector<dyn_var_t> > > > >&
			arrayParamsMap
			);
  void buildBidirectionalConnectionMap(
      std::vector<SegmentDescriptor::SegmentKeyData>& maskVector1,
      std::vector<SegmentDescriptor::SegmentKeyData>& maskVector2,
      std::vector<unsigned int*>& v1_ids, 
      std::vector<unsigned int*>& v2_ids, 
      //    unsigned long long bidirectionalConnectionTargetsMask1,
      //    unsigned long long bidirectionalConnectionTargetsMask2,
      const std::string& myBuf 
      );
  void buildElectricalSynapseConnectionMap(
    std::vector<SegmentDescriptor::SegmentKeyData>& maskVector1,
    std::vector<SegmentDescriptor::SegmentKeyData>& maskVector2,
    std::vector<unsigned int*>& v1_ids, 
    std::vector<unsigned int*>& v2_ids, 
		const std::string& myBuf 
		);
  void buildChemicalSynapseConnectionMap(
    std::vector<SegmentDescriptor::SegmentKeyData>& maskVector1,
    std::vector<SegmentDescriptor::SegmentKeyData>& maskVector2,
    std::vector<unsigned int*>& v1_ids, 
    std::vector<unsigned int*>& v2_ids, 
		const std::string& myBuf 
		);
	bool checkForSpecialCases(FILE* fpF, int sz);
  bool checkForSpecialCases(FILE* fpF, int sz, std::vector<int>& columns_found);
  bool checkForSpecialCases(FILE* fpF, int sz, int& firstcolumn_found);
	bool readBondParams(FILE* fpF);
  bool readAngleParams(FILE* fpF);
  bool readLJParams(FILE* fpF);
  bool readRadii(FILE* fpF);
  bool readTouchTables(FILE* fpF);
  bool readSIParams(FILE* fpF);
  bool readCompartmentVariableTargets(FILE* fpF);//support array-form only BRANCHTYPE 
  bool readCompartmentVariableTargets2(FILE* fpF);// support array-form for any 
  bool readBranchPointTargets(FILE* fpF);
  bool readChannelTargets(FILE* fpF); // obsolete
  bool readChannelTargets2(FILE* fpF); //support array-form for all
  bool readElectricalSynapseTargets(FILE* fpF);
  bool readElectricalSynapseTargets_vector2(FILE* fpF);//support array-form for all
  bool readBidirectionalConnectionTargets(FILE* fpF);
  bool readBidirectionalConnectionTargets_vector(FILE* fpF);//array-form for only BRANCHTYPE
  bool readBidirectionalConnectionTargets_vector2(FILE* fpF); //support array-form for all
  bool readChemicalSynapseTargets(FILE* fpF);
  bool readChemicalSynapseTargets_vector2(FILE* fpF);//support array-form for all
  bool readPreSynapticPointTargets(FILE* fpF);

  unsigned long long readNamedParam(FILE* fpF, std::string name,
                                    std::map<key_size_t, double>& namedParamsMap);

  bool readCompartmentVariableCosts(FILE* fpF);
  bool readChannelCosts(FILE* fpF);
  bool readElectricalSynapseCosts(FILE* fpF);
  bool readBidirectionalConnectionCosts(FILE* fpF);
  bool readChemicalSynapseCosts(FILE* fpF);
  // bool readChannelParams(FILE* fpF);

  bool readModelParams(
      FILE* fpF, const std::string& id,
      std::map<std::string, unsigned long long>& masks,
      std::map<std::string,
               std::map<key_size_t, std::list<std::pair<std::string, dyn_var_t> > > >&
          paramsMap,
      std::map<
          std::string,
          std::map<key_size_t,
                   std::list<std::pair<std::string, std::vector<dyn_var_t> > > > >&
          arrayParamsMap);
  bool readModelParams2(
      FILE* fpF, const std::string& id,
      std::map<std::string, unsigned long long>& masks,
      std::map<std::string,
               std::map<key_size_t, std::list<std::pair<std::string, dyn_var_t> > > >&
          paramsMap,
      std::map<
          std::string,
          std::map<key_size_t,
                   std::list<std::pair<std::string, std::vector<dyn_var_t> > > > >&
          arrayParamsMap);

  unsigned long long resetMask(
      FILE* fpF, std::vector<SegmentDescriptor::SegmentKeyData>& maskVector);
  unsigned long long resetMask(
      FILE* fpF, std::vector<SegmentDescriptor::SegmentKeyData>& maskVector, char* bufS);

  double* _bondK0;
  double* _bondR0;
  int _nBondTypes;

  double* _angleK0;
  double* _angleR0;
  int _nAngleTypes;

  double* _ljEps;
  double* _ljR0;
  int _nLJTypes;

  unsigned long long _radiiMask, _SIParamsMask, _compartmentVariableTargetsMask,
      _channelTargetsMask, _electricalSynapseTargetsMask1,
      _electricalSynapseTargetsMask2, _chemicalSynapseTargetsMask1,
      _chemicalSynapseTargetsMask2;
	unsigned long long _bidirectionalConnectionTargetsMask1,
								_bidirectionalConnectionTargetsMask2;

   //define the distance 'radius' (2nd arg) 
   //   to each compartment having the associated key (1st arg)
  std::map<key_size_t, double> _radiiMap;
   //define mapping one compartment (key in 1st arg)
   //   to another component (key in 1st arg_level2 of the 2nd arg)
   //      the 2nd arg is a map containing the associated parameters (Epsilon,Sigma)
   //      for calculating the force between the 2 compartments
  std::map<key_size_t, std::map<key_size_t, SIParameters> > _SIParamsMap;
   //define mapping one compartment (key in 1st arg)
   //      having what kinds of diffusiable nodes whose names is kept in the list
  std::map<key_size_t, std::list<std::string> > _compartmentVariableTargetsMap;
   //define mapping one compartment (key in 1st arg)
   //      to what kinds of channel/receptors
   //      each channel/receptor is represented as an instance of 'ChannelTarget' class
   //      ChannelTarget class keeps
   //         1. the names of the node
   //         2. the list of names of nodes as providing input to the channel/receptor
   //         3. the list of names of nodes as expecting to receive the output from the channel/receptor
  std::map<key_size_t, std::list<ChannelTarget> > _channelTargetsMap;
   //define mapping from a channel/receptor of a given name (1st arg)
   //     to the mask-key value (2nd arg)
  std::map<std::string, unsigned long long> _channelParamsMasks;
   //define mapping from a channel/receptor of a given name (1st arg)
   //    to a given compartment represented via the key (1st arg_level2)
   //       containing a list of pairs (param,value) 
#ifdef NEWIDEA
	//TEST NEW IDEA
  std::map<std::string,
           std::map<key_size_t, std::list<std::pair<std::string, std::string> > > >
      _channelParamsMapGeneric;
#else
  //map<NaT, map<keymask, list-of-pairs{gbar=value}>
  std::map<std::string,
           std::map<key_size_t, std::list<std::pair<std::string, dyn_var_t> > > >
      _channelParamsMap;
#endif
  //map<NaT, map<keymask, list-of-pairs{gbar,{val1, val2, ..., valn}}>
  std::map<std::string,
           std::map<key_size_t,
                    std::list<std::pair<std::string, std::vector<dyn_var_t> > > > >
      _channelArrayParamsMap;

  std::map<std::string, unsigned long long> _compartmentParamsMasks;
#ifdef NEWIDEA
	//TEST NEW IDEA
  std::map<std::string,
           std::map<key_size_t, std::list<std::pair<std::string, std::string> > > >
      _compartmentParamsMapGeneric;
#else
  std::map<std::string,
           std::map<key_size_t, std::list<std::pair<std::string, dyn_var_t> > > >
      _compartmentParamsMap;
#endif
  std::map<std::string,
           std::map<key_size_t,
                    std::list<std::pair<std::string, std::vector<dyn_var_t> > > > >
      _compartmentArrayParamsMap;
   //define mapping a compartment of a given key (1st arg)
   //    to a second compartment of a given key (1st arg_level2)
   //      each bidirectional connection is represented 'ElectricalSynapseTarget' class
   //      ElectricalSynapseTarget class keeps
   //         1. the names of the node (as the value to nodekind in Layer statement)
   //         2. the list of names of diffusible nodes whose data 
   //             will flow in 2 directions
   //         3. a double _parameter = -1.0 (default) representing the prob. for forming the 
   //                                   bidirectional connection (e.g. electrical synapse
   //                                      or spine_neck + branch's compartment)
  std::map<key_size_t, std::map<key_size_t, std::list<ElectricalSynapseTarget> > > 
      _electricalSynapseTargetsMap;
  std::map<key_size_t, std::map<key_size_t, std::list<BidirectionalConnectionTarget> > > 
      _bidirectionalConnectionTargetsMap;
   //define mapping a compartment of a given key (1st arg)
   //    to a second compartment of a given key (1st arg_level2)
   //      each chemicalsynapse is represented as a list of 'ChemicalSynapseTarget'
   //      an instance of ChemicalSynapseTarget contains
   //          1. a double _parameter = -1.0 (default) representing the prob. for forming the 
   //                                   chemical synapse
   //          2. the map of (names of receptor/channels nodes, a pair of 
   //                          2 list of names
   //                           list 1 = names of nodes as providing input to the channel
   //                           list 2 = name sof nodes as receiving the output from the channel
  std::map<key_size_t, std::map<key_size_t, std::list<ChemicalSynapseTarget> > >
      _chemicalSynapseTargetsMap;
  std::map<std::string, std::string> _preSynapticPointTargetsMap;
  std::map<std::string, std::list<std::string> > _preSynapticPointSynapseMap;
   //define the cost (double) associated with a given dynamic variable
   //   which is used to estimate the complexity in calculating
  std::map<std::string, double> _compartmentVariableCostsMap, _channelCostsMap,
      _electricalSynapseCostsMap, _chemicalSynapseCostsMap;
	std::map<std::string, double> _bidirectionalConnectionCostsMap;

  std::vector<std::vector<SegmentDescriptor::SegmentKeyData> > _touchTableMasks;

  // tell if the information read from the parameter files
  // having the corresponding section
  // e.g. _compartmentVariables == TRUE when  CptParams.par is read
  //      _channels == TRUE when CHEMICAL_SYNAPSE_TARGETS section in SynParams.par is read
  bool _SIParams, _compartmentVariables, _channels, _electricalSynapses,
      _chemicalSynapses;
	bool _bidirectionalConnections;
#ifdef SUPPORT_DEFINING_SPINE_HEAD_N_NECK_VIA_PARAM
  bool _passedInSpineHead, _passedInSpineNeck; 
  unsigned long long _spineHeadMask, _spineNeckMask;
  //std::set<key_size_t>  _spineHeadsMap, _spineNecksMap;
  std::vector<key_size_t>  _spineHeadsMap, _spineNecksMap;
#endif
  bool readCriteriaSpineHead(FILE* fpF);// support array-form for any 
  bool readCriteriaSpineNeck(FILE* fpF);// support array-form for any 
	void buildSpinesMap(
			std::vector<SegmentDescriptor::SegmentKeyData>& maskVector,
			std::vector<unsigned int*>& v_ids, 
      std::vector<key_size_t>& mymap
			//std::istringstream &is
			//const std::string & myBuf
			);

  SegmentDescriptor _segmentDescriptor;
  std::string _currentFName;
};
#endif
