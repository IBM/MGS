// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef LAYERDEFINITIONCONTEXT_H
#define LAYERDEFINITIONCONTEXT_H
#include "Copyright.h"

class NodeSet;
class Grid;
class NodeType;
class NDPairList;

class LayerDefinitionContext
{
 public:
  LayerDefinitionContext() : nodeset(0), grid(0) {}
  NodeSet* nodeset;
  Grid* grid;
};
#endif
