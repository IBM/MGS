// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef PolyConnectorFunctor_H
#define PolyConnectorFunctor_H
#include "Mgs.h"

#include "CG_PolyConnectorFunctorBase.h"
#include "GslContext.h"
#include "NoConnectConnector.h"
#include "GranuleConnector.h"
#include "MgsConnector.h"
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
      void userInitialize(GslContext* CG_c);
      void userExecute(GslContext* CG_c, std::vector<DataItem*>::const_iterator begin, std::vector<DataItem*>::const_iterator end);
      PolyConnectorFunctor();
      virtual ~PolyConnectorFunctor();
      virtual void duplicate(std::unique_ptr<PolyConnectorFunctor>&& dup) const;
      virtual void duplicate(std::unique_ptr<Functor>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_PolyConnectorFunctorBase>&& dup) const;
   private:
      NoConnectConnector _noConnector;
      GranuleConnector _granuleConnector;
      MgsConnector _mgsConnector;
};

#endif
