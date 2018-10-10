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

#ifndef OPENCIRCLELAYOUTFUNCTORTYPE_H
#define OPENCIRCLELAYOUTFUNCTORTYPE_H
#include "Copyright.h"

#include "FunctorType.h"

class OpenCircleLayoutFunctorType : public FunctorType
{
   public:
      OpenCircleLayoutFunctorType();
      void getFunctor(std::unique_ptr<Functor> & r_aptr);
      virtual std::string getName();
      virtual std::string getDescription();
      virtual void getQueriable(std::unique_ptr<InstanceFactoryQueriable>& dup);
      Functor* getFunctor();
      ~OpenCircleLayoutFunctorType();
};
#endif
