// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_declaration_strid_He
#define C_declaration_strid_He
#include "Copyright.h"

#include "C_declaration.h"

class C_declarator;
class C_stride_list;
class LensContext;
class StridesList;
class SyntaxError;

class C_declaration_stride : public C_declaration
{
   public:
      C_declaration_stride(const C_declaration_stride&);
      C_declaration_stride(C_declarator *, C_stride_list *, SyntaxError *);
      virtual C_declaration_stride* duplicate() const;
      virtual ~C_declaration_stride();
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

   private:
      C_declarator* _cDeclarator;
      C_stride_list* _cStrideList;
      StridesList* _stridesList;

};
#endif
