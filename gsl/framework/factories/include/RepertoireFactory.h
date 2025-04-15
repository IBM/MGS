// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef REPERTOIREFACTORY_H
#define REPERTOIREFACTORY_H
#include "Copyright.h"

#include <string>
#include <memory>

class Repertoire;
class LensContext;

class RepertoireFactory
{
   public:
      virtual Repertoire* createRepertoire(std::string const& repName, LensContext* c) = 0;
      virtual void duplicate(std::unique_ptr<RepertoireFactory>& rv) const =0;
      virtual ~RepertoireFactory(){};
};
#endif
