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

#ifndef ConnectNodeSetsFunctor_H
#define ConnectNodeSetsFunctor_H
#include "Lens.h"

#include "CG_ConnectNodeSetsFunctorBase.h"
#include "LensContext.h"
#include <memory>

class NoConnectConnector;
class GranuleConnector;
class LensConnector;

class ConnectNodeSetsFunctor : public CG_ConnectNodeSetsFunctorBase
{
   public:
      void userInitialize(LensContext* CG_c);
      void userExecute(LensContext* CG_c, NodeSet*& source, NodeSet*& destination, Functor*& sampling, Functor*& sourceOutAttr, Functor*& destinationInAttr);
      ConnectNodeSetsFunctor();
      virtual ~ConnectNodeSetsFunctor();
      virtual void duplicate(std::auto_ptr<ConnectNodeSetsFunctor>& dup) const;
      virtual void duplicate(std::auto_ptr<Functor>& dup) const;
      virtual void duplicate(std::auto_ptr<CG_ConnectNodeSetsFunctorBase>& dup) const;
   private:
      NoConnectConnector* _noConnector;
      GranuleConnector* _granuleConnector;
      LensConnector* _lensConnector;
};

#endif
