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

#ifndef C_declaration_grid_granule_H
#define C_declaration_grid_granule_H
#include "Copyright.h"

#include "C_declaration.h"
#include "LensContext.h"
#include <string>
#include <memory>
#include <map>

class C_declarator;
class C_grid_granule;
class SyntaxError;
class GranuleMapper;

class C_declaration_grid_granule : public C_declaration
{
   public:
      C_declaration_grid_granule(const C_declaration_grid_granule&);
      C_declaration_grid_granule(C_declarator *, C_grid_granule *, SyntaxError *);
      virtual C_declaration_grid_granule* duplicate() const;
      virtual ~C_declaration_grid_granule();
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

   private:
      C_declarator* _declarator;
      C_grid_granule* _granuleMapper;
      std::string* _name;
};
#endif
