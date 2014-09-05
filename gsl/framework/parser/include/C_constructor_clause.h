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

#ifndef C_constructor_clause_H
#define C_constructor_clause_H
#include "Copyright.h"

#include <list>
#include "C_complex_functor_clause.h"
class C_parameter_type;
class C_parameter_type_list;
class LensContext;
class SyntaxError;

class C_constructor_clause : public C_complex_functor_clause
{
   public:
      C_constructor_clause(const C_constructor_clause&);
      C_constructor_clause(C_parameter_type_list *, SyntaxError *);
      virtual ~C_constructor_clause();
      virtual C_constructor_clause* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      std::list<C_parameter_type>* getParameterTypeList();
   private:
      C_parameter_type_list* _ptl;

};
#endif
