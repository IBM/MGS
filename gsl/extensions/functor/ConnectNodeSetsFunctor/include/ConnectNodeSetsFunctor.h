// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef ConnectNodeSetsFunctor_H
#define ConnectNodeSetsFunctor_H
#include "Mgs.h"

#include "CG_ConnectNodeSetsFunctorBase.h"
#include "GslContext.h"
#include <memory>

class NoConnectConnector;
class GranuleConnector;
class MgsConnector;

class ConnectNodeSetsFunctor : public CG_ConnectNodeSetsFunctorBase
{
   public:
      void userInitialize(GslContext* CG_c);
      void userExecute(GslContext* CG_c, NodeSet*& source, NodeSet*& destination, Functor*& sampling, Functor*& sourceOutAttr, Functor*& destinationInAttr);
      ConnectNodeSetsFunctor();
      virtual ~ConnectNodeSetsFunctor();
      virtual void duplicate(std::unique_ptr<ConnectNodeSetsFunctor>&& dup) const;
      virtual void duplicate(std::unique_ptr<Functor>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_ConnectNodeSetsFunctorBase>&& dup) const;
   private:
      NoConnectConnector* _noConnector;
      GranuleConnector* _granuleConnector;
      MgsConnector* _mgsConnector;
};

#endif
