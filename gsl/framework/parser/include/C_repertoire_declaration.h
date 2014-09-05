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

#ifndef C_repertoire_declaration_H
#define C_repertoire_declaration_H
#include "Copyright.h"

#include <string>
#include "C_production.h"

class C_declarator;
class LensContext;
class SyntaxError;

class C_repertoire_declaration : public C_production
{
   public:
      C_repertoire_declaration(const C_repertoire_declaration&);
      C_repertoire_declaration(C_declarator *type, C_declarator *name, 
			       SyntaxError *);
      virtual ~C_repertoire_declaration();
      virtual C_repertoire_declaration* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      std::string const getType();
      std::string const getName();

   private:
      C_declarator *_typeName;
      C_declarator *_instanceName;
      std::string _type; // For Empty  
      std::string _name; // For Empty  
};
#endif
