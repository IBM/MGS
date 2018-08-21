// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef C_matrix_type_specifier_H
#define C_matrix_type_specifier_H
#include "Copyright.h"

#include "C_production.h"

class C_type_specifier;
class LensContext;
class SyntaxError;

class C_matrix_type_specifier: public C_production
{
   public:
      C_matrix_type_specifier(const C_matrix_type_specifier&);
      C_matrix_type_specifier(C_type_specifier *, SyntaxError *);
      virtual ~C_matrix_type_specifier();
      virtual C_matrix_type_specifier* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      C_type_specifier* getTypeSpecifier() const;

   private:
      C_type_specifier* _typeSpecifier;
};
#endif
