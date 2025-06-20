// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
