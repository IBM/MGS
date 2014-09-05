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

#ifndef INATTRDEFAULTFUNCTORTYPE_H
#define INATTRDEFAULTFUNCTORTYPE_H
#include "Copyright.h"

#include "FunctorType.h"

class InAttrDefaultFunctorType : public FunctorType
{
   public:
      InAttrDefaultFunctorType();
      void getFunctor(std::auto_ptr<Functor> & r_aptr);
      virtual std::string getName();
      virtual std::string getDescription();
      virtual void getQueriable(std::auto_ptr<InstanceFactoryQueriable>& dup);
      Functor* getFunctor();
      ~InAttrDefaultFunctorType();
};
#endif
