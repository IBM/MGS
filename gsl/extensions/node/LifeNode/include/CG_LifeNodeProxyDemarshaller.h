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

#ifndef CG_LifeNodeProxyDemarshaller_H
#define CG_LifeNodeProxyDemarshaller_H

#include "Lens.h"
#ifdef HAVE_MPI
#include "CG_LifeNodeProxyDemarshaller.h"
#include "StructDemarshallerBase.h"

class CG_LifeNodeProxy;

class CG_LifeNodeProxyDemarshaller : public StructDemarshallerBase
{
   public:
      CG_LifeNodeProxyDemarshaller()
      : _proxy(0)
      {
      }
      CG_LifeNodeProxyDemarshaller(CG_LifeNodeProxy* p)
      : _proxy(p)
      {
      }
      virtual void setDestination(CG_LifeNodeProxy *proxy)=0;
      virtual ~CG_LifeNodeProxyDemarshaller()
      {
      }
   protected:
      CG_LifeNodeProxy* _proxy;
};

#endif
#endif
