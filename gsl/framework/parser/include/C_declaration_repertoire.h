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

#ifndef C_declaration_repertoire_H
#define C_declaration_repertoire_H
#include "Copyright.h"

#include "C_declaration.h"
#include <memory>
#include <map>

class C_repertoire_declaration;
class LensContext;
class SyntaxError;

class C_declaration_repertoire : public C_declaration
{
   public:
      C_declaration_repertoire(const C_declaration_repertoire&);
      C_declaration_repertoire(C_repertoire_declaration *, SyntaxError *);
      virtual C_declaration_repertoire* duplicate() const;
      virtual ~C_declaration_repertoire();
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

   private:
      C_repertoire_declaration* _repertoireDeclaration;
};
#endif
