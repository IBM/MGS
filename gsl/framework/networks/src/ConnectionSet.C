// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "ConnectionSet.h"
#include "GridLayerDescriptor.h"
#include "Edge.h"
#include "Node.h"
#include <stdio.h>
#include <stdlib.h>

#include <sstream>

ConnectionSet::ConnectionSet(GridLayerDescriptor* pre, GridLayerDescriptor* post)
{
   _pre = pre;
   _post = post;
   std::ostringstream name;
   std::string preName = _pre->getName();
   std::string postName=  _post->getName();
   name<<"["<<preName<<"->"<<postName<<"]";
   _name = name.str();
}

ConnectionSet::ConnectionSet(ConnectionSet* cs)
: std::vector<Edge*>(*cs)
{
   _pre = cs->_pre;
   _post = cs->_post;
   _name = cs->_name;
}

GridLayerDescriptor* ConnectionSet::getPrePtr()
{
   return _pre;
}

GridLayerDescriptor* ConnectionSet::getPostPtr()
{
   return _post;
}

void ConnectionSet::addConnection(Edge* e)
{
   if (e->getPreNode()->getGridLayerDescriptor() != _pre) {
      std::cerr<<"Edge with invalid pre node added to Connection Set!"<<std::endl;
      exit(-1);
   }
   if (e->getPostNode()->getGridLayerDescriptor() != _post) {
      std::cerr<<"Edge with invalid post node added to Connection Set!"<<std::endl;
      exit(-1);
   }
   this->push_back(e);
}

void ConnectionSet::addEdge(Edge* e)
{
   this->push_back(e);
}

std::string ConnectionSet::getName()
{
   return _name;
}

ConnectionSet::~ConnectionSet()
{
   // ** the edges are owned by the comp categories and shouldn't(???) be deleted here***
}
