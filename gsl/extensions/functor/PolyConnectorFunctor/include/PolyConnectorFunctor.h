// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef PolyConnectorFunctor_H
#define PolyConnectorFunctor_H
#include "Lens.h"

#include "CG_PolyConnectorFunctorBase.h"
#include "LensContext.h"
#include "NoConnectConnector.h"
#include "GranuleConnector.h"
#include "LensConnector.h"
#include <memory>

class Constant;
class Variable;
class NodeSet;
class EdgeSet;
class NDPairList;
class Simulation;

class PolyConnectorFunctor : public CG_PolyConnectorFunctorBase
{
   public:
      void userInitialize(LensContext* CG_c);
      void userExecute(LensContext* CG_c, std::vector<DataItem*>::const_iterator begin, std::vector<DataItem*>::const_iterator end);
      PolyConnectorFunctor();
      virtual ~PolyConnectorFunctor();
      virtual void duplicate(std::auto_ptr<PolyConnectorFunctor>& dup) const;
      virtual void duplicate(std::auto_ptr<Functor>& dup) const;
      virtual void duplicate(std::auto_ptr<CG_PolyConnectorFunctorBase>& dup) const;
   private:
      NoConnectConnector _noConnector;
      GranuleConnector _granuleConnector;
      LensConnector _lensConnector;
};

#endif
