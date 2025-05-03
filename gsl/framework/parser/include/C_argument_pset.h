// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_ARGUMENT_PSET_H
#define C_ARGUMENT_PSET_H
#include "Copyright.h"

#include <string>
#include <memory>
#include "C_argument.h"

class C_parameter_type_pair;
class C_ndpair_clause_list;
class GslContext;
class DataItem;
class ParameterSetDataItem;
class SyntaxError;

class C_argument_pset: public C_argument
{
   public:
      C_argument_pset(const C_argument_pset&);
      C_argument_pset(C_parameter_type_pair *, C_ndpair_clause_list *, 
		      SyntaxError *);
      virtual ~C_argument_pset();
      virtual C_argument_pset* duplicate() const;
      virtual void internalExecute(GslContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

      // accessors
      C_parameter_type_pair *getParameter_type_pair() const { 
	 return _parm_type_pair; 
      }
      C_ndpair_clause_list *getNdpair_clause_list() const { 
	 return _ndp_clause_list; 
      }
      DataItem *getArgumentDataItem() const;

   private:
      C_parameter_type_pair* _parm_type_pair;
      C_ndpair_clause_list* _ndp_clause_list;
      ParameterSetDataItem* _parameter_set_DI;
};
#endif
