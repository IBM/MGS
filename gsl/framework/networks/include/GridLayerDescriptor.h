// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef GRIDLAYERDESCRIPTOR_H
#define GRIDLAYERDESCRIPTOR_H
#include "Copyright.h"

#include "NDPairList.h"

#include <memory>
#include <vector>
#include <string>

class NodeAccessor;
class NodeType;
class Grid;

// SGC comments [begin]
// The layer descriptor describes the density of each position.
// It is trivial if the density is uniform, if not, the density vector is
// used to figure out the density of the node cooedinate.
// If the size of the density vector is smaller than the total number of
// node coordinates, than the density vector is wrapped around the total
// number of node coordinates (modulus).
// SGC comments [end]

class GridLayerDescriptor
{
   friend class Grid;

   public:
      std::string getModelName();

      int getDensity(const std::vector<int>& coords);
      int getDensity(int nodeIndex);
      unsigned getMaxDensity();
      unsigned getMinDensity();

      virtual ~GridLayerDescriptor();

      const std::string& getName() const {
	 return _name;
      }

      NodeType *getNodeType() {
	 return _nt;
      }

      Grid* getGrid() const {
	 return _grid;
      }

      int isUniform() {
	 return _uniformDensity;
      }

      NodeAccessor* getNodeAccessor() {
	 return _nodeAccessor;
      }

      const NDPairList& getNDPList() const {
	 return _ndpList;
      }

      unsigned getGranuleMapperIndex() const {
	 return _granuleMapperIndex;
      }

      void replaceDensityVector(unsigned* replacement, int uniform);

   private:
      GridLayerDescriptor(Grid* grid, const std::vector<int>& densityVector, 
			  std::string name, NodeType* nt, 
			  const NDPairList& ndpl, unsigned granuleMapperIndex);

      void setNodeAccessor(std::unique_ptr<NodeAccessor>&);

	  //NOTE: Example Layer declared in GLS
	  //Layer(layer_name, HodgkinHuxleyVoltage, tissueFunctor("Layout", <nodekind="Branches"),
	  //        <nodekind="Branches")
      Grid* _grid;
      NodeAccessor* _nodeAccessor;
      std::string _name; //e.g. would be 'layer_name'
      NodeType* _nt; // point to LifeNodeCompCategory, HodgkinHuxleyVoltageCompCategory
      std::vector<int> _densityVector;
      int _uniformDensity;
      NDPairList _ndpList;// this is the information in the last argument passed to 
	     //...Layer statement, e.g. <nodekind="Branches">
      unsigned _granuleMapperIndex;
};

#endif
