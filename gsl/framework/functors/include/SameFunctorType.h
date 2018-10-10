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

#ifndef SAMEFUNCTORTYPE_H
#define SAMEFUNCTORTYPE_H
#include "Copyright.h"

#include "FunctorType.h"

class SameFunctorType : public FunctorType
{
   public:
      SameFunctorType();
      void getFunctor(std::unique_ptr<Functor> & r_aptr);
      virtual std::string getName();
      virtual std::string getDescription();
      virtual void getQueriable(std::unique_ptr<InstanceFactoryQueriable>& dup);
      Functor* getFunctor();
      ~SameFunctorType();
};
#endif
