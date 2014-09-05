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
      virtual void duplicate(std::auto_ptr<Functor> &fap) const;
      virtual ~ConnectSets2Functor();
   protected:
      virtual void doInitialize(LensContext *c, 
				const std::vector<DataItem*>& args);
      virtual void doExecute(LensContext *c, 
			     const std::vector<DataItem*>& args, 
			     std::auto_ptr<DataItem>& rvalue);
   private:
      NoConnectConnector* _noConnector;
      GranuleConnector* _granuleConnector;
      LensConnector* _lensConnector;
};
#endif
