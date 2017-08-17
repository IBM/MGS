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

#ifndef C_GRID_GRANULE_H
#define C_GRID_GRANULE_H
#include "Copyright.h"

#include <string>
#include <memory>
#include "C_production.h"
#include <vector>

class LensContext;
class GranuleMapper;
class ConnectionIncrement;
class C_declarator;

class C_grid_granule : public C_production
{
   public:
      C_grid_granule(C_declarator*, SyntaxError* error);
      C_grid_granule(const C_grid_granule&);
      virtual ~C_grid_granule();
      virtual C_grid_granule* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      virtual GranuleMapper* getGranuleMapper() {return _granuleMapper;}

   private:
      C_declarator* _declarator;
      GranuleMapper* _granuleMapper;
};
#endif
