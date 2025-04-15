// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_parameter_type_list_H
#define C_parameter_type_list_H
#include "Copyright.h"

#include <list>
#include "C_production.h"

class C_parameter_type;
class LensContext;
class SyntaxError;

class C_parameter_type_list : public C_production
{
   public:
      C_parameter_type_list(const C_parameter_type_list&);
      C_parameter_type_list(SyntaxError *);
      C_parameter_type_list(C_parameter_type *, SyntaxError *);
      C_parameter_type_list(C_parameter_type_list *, C_parameter_type *, 
			    SyntaxError *);
      std::list<C_parameter_type>* releaseList();
      virtual ~C_parameter_type_list ();
      virtual C_parameter_type_list* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      std::list<C_parameter_type>* getList() {
	 return _list;
      }

   private:
      std::list<C_parameter_type>* _list;
};
#endif
