// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "GranuleMapperBase.h"
#include <cassert>

GranuleMapperBase::GranuleMapperBase() : _index(0)
{
}

void GranuleMapperBase::setGraphId(unsigned& current)
{
   std::vector<Granule>::iterator it, end = _granules.end();
   for (it = _granules.begin(); it != end; ++it) {
      it->setGraphId(current);
   }
}

void GranuleMapperBase::initializeGraph(Graph* graph)
{
   std::vector<Granule>::iterator it, end = _granules.end();
   for (it = _granules.begin(); it != end; ++it) {
      it->initializeGraph(graph);
   }
}

void GranuleMapperBase::setGlobalGranuleIds(unsigned& id)
{
   std::vector<Granule>::iterator it, end = _granules.end();
   for (it = _granules.begin(); it != end; ++it) {
      it->setPartitionId(id);
      it->setGlobalGranuleId(id++);
   }
}

GranuleMapperBase::~GranuleMapperBase()
{
}

