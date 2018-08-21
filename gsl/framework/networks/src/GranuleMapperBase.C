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

