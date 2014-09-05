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

#ifndef CONNECTORFUNCTOR_H
#define CONNECTORFUNCTOR_H
#include "Copyright.h"

#include "Functor.h"
class ConnectorFunctor : public Functor
{
   public:
      const char * getCategory();
      static const char* _category;
};
#endif
