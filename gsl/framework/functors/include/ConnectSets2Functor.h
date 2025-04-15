// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef _CONNECTSETS2FUNCTOR_H_
#define _CONNECTSETS2FUNCTOR_H_
#include "Copyright.h"

#include "ConnectorFunctor.h"
#include <memory>
#include <list>
#include <vector>
class DataItem;
class LensContext;
class NoConnectConnector;
class GranuleConnector;
class LensConnector;

class ConnectSets2Functor: public ConnectorFunctor
{
   public:
      ConnectSets2Functor();
      virtual void duplicate(std::unique_ptr<Functor>&& fap) const;
      virtual ~ConnectSets2Functor();
   protected:
      virtual void doInitialize(LensContext *c, 
				const std::vector<DataItem*>& args);
      virtual void doExecute(LensContext *c, 
			     const std::vector<DataItem*>& args, 
			     std::unique_ptr<DataItem>& rvalue);
   private:
      NoConnectConnector* _noConnector;
      GranuleConnector* _granuleConnector;
      LensConnector* _lensConnector;
};
#endif
