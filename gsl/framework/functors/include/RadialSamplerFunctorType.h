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

#ifndef RADIALSAMPLERFUNCTORTYPE_H
#define RADIALSAMPLERFUNCTORTYPE_H
#include "Copyright.h"

#include "FunctorType.h"

class RadialSamplerFunctorType : public FunctorType
{
   public:
      RadialSamplerFunctorType();
      void getFunctor(std::auto_ptr<Functor> & r_aptr);
      virtual std::string getName();
      virtual std::string getDescription();
      virtual void getQueriable(std::auto_ptr<InstanceFactoryQueriable>& dup);
      Functor* getFunctor();
      ~RadialSamplerFunctorType();
};
#endif
