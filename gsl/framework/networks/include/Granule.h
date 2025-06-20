// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
	 //_old_partitionId = _partitionId;
	 _partitionId = partitionId; 
      }
      unsigned getPartitionId() const;
      //void  getOldPartitionId(unsigned partitionId) {
      //   return _old_partitionId;
      //}

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
      unsigned _partitionId;  //initially, this is the same as _old_partitionId
        // after sim->setGraph(), it is equals to the MPI-rank on which the nodes associated with 
	// such Granule should be created
      //unsigned _old_partitionId;
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
