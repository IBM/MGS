// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifdef HAVE_MPI
#include <mpi.h>
#endif
#include "Repertoire.h"
#include "Grid.h"
#include "ConnectionSet.h"
#include "GridLayerDescriptor.h"
#include "NodeDescriptor.h"
#include "Edge.h"
#include <stdio.h>
#include <stdlib.h>

                                 //string description
Repertoire::Repertoire(std::string name)
: _parentRepertoire(0), _grid(0), _name(name)
{
}

const std::string& Repertoire::getName() const
{
   return _name;
}

void Repertoire::setName(const std::string& str)
{
   _name = str;
}

std::list<Repertoire*> const & Repertoire::getSubRepertoires() const
{
   return _subRepertoires;
}

Repertoire* Repertoire::getSubRepertoire(std::string repName)
{
   Repertoire* rval = 0;
   std::list<Repertoire*>::iterator iter = _subRepertoires.begin(), 
      end = _subRepertoires.end();
   for (;iter!=end;++iter) {
      if ((*iter)->getName() == repName) {
         rval = (*iter);
         break;
      }
   }
   if (iter == end) {
      std::cerr << "Subrepertoire " << repName << " not found in repertoire "
		<< _name << "!" << std::endl;
      exit(-1);
   }
   return rval;
}


Repertoire* Repertoire::getParentRepertoire() const
{
   return _parentRepertoire;
}


Grid* Repertoire::setGrid(std::vector<int> size)
{
   if ((_subRepertoires.size() == 0) && !(isGridRepertoire())) {
      _grid = new Grid(size);
      _grid->setParentRepertoire(this);
   }
   else if (_subRepertoires.size()>0) {
      std::cerr<<"Cannot set grid on composite repertoire!"<<std::endl;
   }
   else std::cerr<<"Cannot set grid on repertoire.  Grid already set!"<<std::endl;
   return _grid;
}


Grid* Repertoire::getGrid() const
{
   return _grid;
}

void Repertoire::addSubRepertoire(std::auto_ptr<Repertoire> & rep)
{
   if (_grid != 0)
      std::cerr<<"Cannot add Sub-Repertoire to Grid Repertoire!"<<std::endl;
   else if (rep->getParentRepertoire() != 0)
      std::cerr<<"Cannot add specified Sub-Repertoire whose Parent is already specified!"<<std::endl;
   else {
      Repertoire* ptrRep = rep.release();
      _subRepertoires.push_back(ptrRep);
      ptrRep->setParentRepertoire(this);
   }
}

void Repertoire::setParentRepertoire(Repertoire* rep)
{
   if (rep->isGridRepertoire())
      std::cerr<<"Cannot set Grid Repertoire as Parent Repertoire!"<<std::endl;
   else
      _parentRepertoire = rep;
}

ConnectionSet* Repertoire::getConnectionSet(GridLayerDescriptor* prePtr, GridLayerDescriptor* postPtr)
{
   ConnectionSet* rval = 0;
   // check if specified ConnectionSet is already in map entry's list
   std::map<GridLayerDescriptor*, std::list<ConnectionSet*> >::iterator entry = _connectionSetMap.find(prePtr);
   if(entry != _connectionSetMap.end()) {
      std::list<ConnectionSet*>::iterator end = (*entry).second.end();
      for (std::list<ConnectionSet*>::iterator iter = (*entry).second.begin(); iter != end; iter++) {
         if ((*iter)->getPostPtr() == postPtr) {
            rval = (*iter);
         }
      }
   }
   if (!rval) {
      std::cerr<<"No ConnectionSet found for specified pre and post GridLayerDescriptor!"<<std::endl;
      exit(-1);
   }
   return rval;
}

std::list<ConnectionSet*> const & Repertoire::getConnectionSetList(GridLayerDescriptor* prePtr)
{
   std::map<GridLayerDescriptor*, std::list<ConnectionSet*> >::iterator entry = _connectionSetMap.find(prePtr);
   if(entry == _connectionSetMap.end()) {
      std::cerr<<"No Connection Set found for specified pre GridLayerDescriptor!"<<std::endl;
      exit(-1);
   }
   return (*entry).second;
}

std::map<GridLayerDescriptor*, std::list<ConnectionSet*> > const & Repertoire::getConnectionSetMap()
{
   return _connectionSetMap;
}

void Repertoire::addConnection(Edge* e)
{
   NodeDescriptor* n = e->getPreNode();

   // n==0 indicates that the edge has no relational data 
   // (i.e., the command line memory-saving option -erd is set to 0)
   if (n==0) return; 


   GridLayerDescriptor* prePtr = n->getGridLayerDescriptor();
   n =  e->getPostNode();
   GridLayerDescriptor* postPtr = n->getGridLayerDescriptor();

   // check if prePtr points to a GridLayerDescriptor already in map
   std::map<GridLayerDescriptor*, std::list<ConnectionSet*> >::iterator entry = _connectionSetMap.find(prePtr);
   if (entry != _connectionSetMap.end()) {
      // check if specified ConnectionSet is already in map entry's list
      bool post_in_map = false;
      std::list<ConnectionSet*>::iterator end = (*entry).second.end();
      for (std::list<ConnectionSet*>::iterator iter = (*entry).second.begin(); iter != end; iter++) {
         if ((*iter)->getPostPtr() == postPtr) {
            post_in_map = true;
            (*iter)->addEdge(e);
            break;
         }
      }
      if (!post_in_map) {
         ConnectionSet* ptrConnectionSet;
         ptrConnectionSet = new ConnectionSet(prePtr, postPtr);
         ptrConnectionSet->addEdge(e);
         _connectionSetMap[prePtr].push_back(ptrConnectionSet);
      }
   }
   else {
      // add a list of connection sets and a new Connection Set to map
      std::list<ConnectionSet*> newList;
      ConnectionSet* ptrConnectionSet;
      ptrConnectionSet = new ConnectionSet(prePtr, postPtr);
      ptrConnectionSet->addEdge(e);
      newList.push_back(ptrConnectionSet);
      _connectionSetMap[prePtr] = newList;
   }
}


bool Repertoire::isGridRepertoire() const
{
   return (_grid!=0);
}


Repertoire::~Repertoire()
{
   delete _grid;
   std::list<Repertoire*>::iterator end = _subRepertoires.end();
   for (std::list<Repertoire*>::iterator iter = _subRepertoires.begin(); iter != end; iter++)
      delete (*iter);
   std::map<GridLayerDescriptor*, std::list <ConnectionSet*> >::iterator end2 = _connectionSetMap.end();
   for (std::map<GridLayerDescriptor*, std::list <ConnectionSet*> >::iterator iter = _connectionSetMap.begin(); iter != end2; iter++) {
      std::list<ConnectionSet*>::iterator end3 = (*iter).second.end();
      for (std::list<ConnectionSet*>::iterator iter2 = (*iter).second.begin(); iter2 != end3; iter2++)
         delete (*iter2);
   }
}

std::ostream& operator<<(std::ostream& os, const Repertoire& inp)
{
   os << inp.getName();
   return os;
}
