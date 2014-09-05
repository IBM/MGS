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
