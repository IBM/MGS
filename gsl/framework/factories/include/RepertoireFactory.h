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
      virtual void duplicate(std::auto_ptr<RepertoireFactory>& rv) const =0;
      virtual ~RepertoireFactory(){};
};
#endif
