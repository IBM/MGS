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

#ifndef C_declaration_publisher_H
#define C_declaration_publisher_H
#include "Copyright.h"

#include "C_declaration.h"
#include <memory>

class C_declarator;
class C_query_path;
class LensContext;
class SyntaxError;

class C_declaration_publisher : public C_declaration
{
   public:
      C_declaration_publisher(const C_declaration_publisher&);
      C_declaration_publisher(C_declarator *, C_query_path *, SyntaxError *);
      virtual C_declaration_publisher* duplicate() const;
      virtual ~C_declaration_publisher();
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

   private:
      C_declarator* _declarator;
      C_query_path* _query_path;
};
#endif
