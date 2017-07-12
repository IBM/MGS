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

#ifndef C_RETURN_CLAUSE_H
#define C_RETURN_CLAUSE_H
#include "Copyright.h"

#include <list>
#include "C_complex_functor_clause.h"
class C_parameter_type;
class C_parameter_type_list;
class LensContext;
class SyntaxError;

class C_return_clause: public C_complex_functor_clause
{
   public:
      C_return_clause(const C_return_clause&);
      C_return_clause(C_parameter_type_list *, SyntaxError *);
      virtual ~C_return_clause();
      virtual C_return_clause* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      std::list<C_parameter_type>* getParameterTypeList();

   private:
      C_parameter_type_list *_ptl;

};
#endif
