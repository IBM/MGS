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

#ifndef REFPTGENFUNCTOR_H
#define REFPTGENFUNCTOR_H
#include "Copyright.h"

#include "Functor.h"
class RefPtGenFunctor : public Functor
{
   public:
      virtual  const char * getCategory();
      static const char* _category;
};
#endif
