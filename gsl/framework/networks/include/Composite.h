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

#ifndef COMPOSITE_H
#define COMPOSITE_H
#include "Copyright.h"

class Repertoire;

class Composite
{

   public:
      Composite();
      Repertoire* getRepertoire() const;
      void setRepertoire(Repertoire*);
      ~Composite();

   private:
      Repertoire* _repertoire;
};
#endif
