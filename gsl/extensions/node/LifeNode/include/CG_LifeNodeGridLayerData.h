// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM
//
// BCM-YKT-12-03-2018
//
//  (C) Copyright IBM Corp. 2005-2018  All rights reserved   .
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef CG_LifeNodeGridLayerData_H
#define CG_LifeNodeGridLayerData_H

#include "Lens.h"
#if defined(HAVE_MPI)
#include "CG_LifeNodeProxy.h"
#endif
#include "Grid.h"
#include "GridLayerData.h"
#include "NodeInstanceAccessor.h"
#if defined(HAVE_MPI)
#include "ShallowArray.h"
#endif

class CG_LifeNodeCompCategory;
class GridLayerDescriptor;
class LifeNode;
class NodeRelationalDataUnit;

class CG_LifeNodeGridLayerData : public GridLayerData
{
   public:
      CG_LifeNodeGridLayerData(CG_LifeNodeCompCategory* compCategory, GridLayerDescriptor* gridLayerDescriptor, int gridLayerIndex);
      NodeInstanceAccessor* getNodeInstanceAccessors();
      virtual ~CG_LifeNodeGridLayerData();
      NodeInstanceAccessor* _nodeInstanceAccessors;
};

#endif
