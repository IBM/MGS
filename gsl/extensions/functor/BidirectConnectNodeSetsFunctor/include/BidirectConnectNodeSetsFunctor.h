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

#ifndef BidirectConnectNodeSetsFunctor_H
#define BidirectConnectNodeSetsFunctor_H
#include "Lens.h"

#include "CG_BidirectConnectNodeSetsFunctorBase.h"
#include "LensContext.h"
#include <memory>

class NoConnectConnector;
class GranuleConnector;
class LensConnector;

class BidirectConnectNodeSetsFunctor : public CG_BidirectConnectNodeSetsFunctorBase
{
   public:
      void userInitialize(LensContext* CG_c);
      void userExecute(LensContext* CG_c, NodeSet*& source, NodeSet*& destination, Functor*& sampling, Functor*& sourceOutAttr, Functor*& destinationInAttr, Functor*& destinationOutAttr, Functor*& sourceInAttr);
      BidirectConnectNodeSetsFunctor();
      virtual ~BidirectConnectNodeSetsFunctor();
      virtual void duplicate(std::unique_ptr<BidirectConnectNodeSetsFunctor>& dup) const;
      virtual void duplicate(std::unique_ptr<Functor>& dup) const;
      virtual void duplicate(std::unique_ptr<CG_BidirectConnectNodeSetsFunctorBase>& dup) const;
   private:
      NoConnectConnector* _noConnector;
      GranuleConnector* _granuleConnector;
      LensConnector* _lensConnector;
};

#endif
