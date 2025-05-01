// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef ConnectNodeSetsByVolumeFunctor_H
#define ConnectNodeSetsByVolumeFunctor_H

#include "Mgs.h"
#include "CG_ConnectNodeSetsByVolumeFunctorBase.h"
#include "LensContext.h"
#include <memory>

class NoConnectConnector;
class GranuleConnector;
class LensConnector;

class ConnectNodeSetsByVolumeFunctor : public CG_ConnectNodeSetsByVolumeFunctorBase
{
   public:
      void userInitialize(LensContext* CG_c);
      void userExecute(LensContext* CG_c, NodeSet*& source, NodeSet*& destination, CustomString& center, float& radius, float& scale, Functor*& sourceOutAttr, Functor*& destinationInAttr);
      ConnectNodeSetsByVolumeFunctor();
      virtual ~ConnectNodeSetsByVolumeFunctor();
      virtual void duplicate(std::unique_ptr<ConnectNodeSetsByVolumeFunctor>&& dup) const;
      virtual void duplicate(std::unique_ptr<Functor>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_ConnectNodeSetsByVolumeFunctorBase>&& dup) const;
   private:
      NoConnectConnector* _noConnector;
      GranuleConnector* _granuleConnector;
      LensConnector* _lensConnector;
};

#endif
