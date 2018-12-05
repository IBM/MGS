// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef CG_LifeNodeFactory_H
#define CG_LifeNodeFactory_H

#include "Lens.h"
class NDPairList;
class NodeType;
class Simulation;

class CG_LifeNodeFactory
{
   public:
      CG_LifeNodeFactory();
      ~CG_LifeNodeFactory();
};

extern "C"
{
   NodeType* CG_LifeNodeFactoryFunction(Simulation& s, const NDPairList& ndpList);
}

#endif
