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

#ifndef _EACHSRCPROPDSTFUNCTORTYPE_H
#define _EACHSRCPROPDSTFUNCTORTYPE_H
#include "Copyright.h"

#include "FunctorType.h"

class EachSrcPropDstFunctorType : public FunctorType
{
   public:
      EachSrcPropDstFunctorType();
      void getFunctor(std::auto_ptr<Functor> & r_aptr);
      virtual std::string getName();
      virtual std::string getDescription();
      virtual void getQueriable(std::auto_ptr<InstanceFactoryQueriable>& dup);
      Functor* getFunctor();
      ~EachSrcPropDstFunctorType();
};
#endif
