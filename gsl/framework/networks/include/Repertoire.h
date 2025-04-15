// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef REPERTOIRE_H_
#define REPERTOIRE_H_
#include "Copyright.h"

#include <list>
#include <map>
#include <string>
#include <vector>
#include <memory>
#include <iostream>

class Edge;
class Grid;
class ConnectionSet;
class GridLayerDescriptor;

class Repertoire
{
   public:

      Repertoire(std::string name = "");
      const std::string& getName() const;
      void setName(const std::string& str);
      Grid* getGrid() const;
      Grid* setGrid(std::vector<int> size);
      std::list<Repertoire*> const & getSubRepertoires() const;
      Repertoire* getSubRepertoire(std::string repName);
      Repertoire* getParentRepertoire() const;
      void addSubRepertoire(std::unique_ptr<Repertoire> &);
      ConnectionSet* getConnectionSet(GridLayerDescriptor* prePtr, 
				      GridLayerDescriptor* postPtr);
      std::list<ConnectionSet*> const & getConnectionSetList(
	 GridLayerDescriptor* prePtr);
      std::map<GridLayerDescriptor*, 
	       std::list<ConnectionSet*> > const & getConnectionSetMap();
      void addConnection(Edge*);
      bool isGridRepertoire() const;
      virtual ~Repertoire();

   protected:
      void addSubGrids(std::list<Grid*> const &);

   private:
      // setParentRepertoire() is private because it is called by 
      // addSubRepertoire() only.
      // Therefore, initialization should make use of addSubRepertoire() 
      // only. JK.
      void setParentRepertoire(Repertoire* rep);
      Repertoire* _parentRepertoire;
      std::list<Repertoire*> _subRepertoires;
      std::map<GridLayerDescriptor*, 
	       std::list<ConnectionSet*> > _connectionSetMap;
      Grid* _grid;
      std::string _name;
};

extern std::ostream& operator<<(std::ostream& os, const Repertoire& inp);
#endif
