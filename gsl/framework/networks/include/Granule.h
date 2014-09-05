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

#ifndef Granule_H
#define Granule_H
#include "Copyright.h"

#include "GranuleConnection.h"
#include "GraphConnection.h"
#include "ConnectionIncrement.h"
#include "RNG.h"

#include <set>
#include <map>
#include <iostream>
#include <vector>

class Graph;

class Granule
{
   public:
      Granule();
      ~Granule();
      std::vector<double>& getModifiableGranuleCoordinates() {
	return _granuleCoordinates;
      }

      void setGlobalGranuleId(unsigned id) {
	 _globalGranuleId = id;
      }
      unsigned getGlobalGranuleId() const {
	 return _globalGranuleId;
      }
      void setGraphId(unsigned& current);
      unsigned getGraphId() const {
	 return _graphId;
      }
      void  setPartitionId(unsigned partitionId) {
	 _partitionId = partitionId; 
      }
      unsigned getPartitionId() const;

      std::set<GraphConnection> const & getGraphConnections() const {
	 return _graphConnections;
      }
      std::set<GraphConnection> & getModifiableGraphConnections() {
	 return _graphConnections;
      }

#if 0
      ConnectionIncrement getComputeCost(std::string phaseName) {
	 return _computeCost[phaseName];
      }
#else
      ConnectionIncrement getComputeCost() {
	 return *_computeCost;
      }
#endif

#if 0
      void addComputeCost(const int density, std::map<std::string, ConnectionIncrement>* cost) {
         std::map<std::string, ConnectionIncrement>::iterator iter = cost->begin(), end = cost->end();
         for (;iter != end; ++iter) {
            _computeCost[(*iter).first] += (*iter).second;
         }
      }
#else
      void addComputeCost(const int density, ConnectionIncrement* cost) {
	 _computeCost->_computationTime += cost->_computationTime;
	 _computeCost->_memoryBytes += cost->_memoryBytes;
	 _computeCost->_communicationBytes += cost->_communicationBytes;
      }
#endif
      void addConnection(Granule* post, float weight = 1.0);
      void addGraphConnection(unsigned graphId, float weight = 1.0);
      void setDepends(Granule* depends);
      Granule* depends() {return _depends;}
      void initializeGraph(Graph* graph);

   private:
      std::vector<double> _granuleCoordinates;
      unsigned _globalGranuleId;
      unsigned _graphId;
      unsigned _partitionId;
      ConnectionIncrement* _computeCost;
//      std::map<std::string, ConnectionIncrement> _computeCost;
      // _depends is used to figure out the graphId if this granule 
      // can't be separated from another granule, it'll have the same 
      // graphId --sgc
      Granule* _depends;
      unsigned _granuleSeed;               // added by Jizhu Lu on 05/05/2006
      std::set<GranuleConnection> _connections;
      std::set<GraphConnection> _graphConnections;
};


extern std::ostream& operator<<(std::ostream& os, const Granule& inp);
extern std::istream& operator>>(std::istream& is, Granule& inp);

#endif
