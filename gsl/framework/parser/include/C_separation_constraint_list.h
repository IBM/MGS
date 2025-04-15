// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_separation_constraint_list_H
#define C_separation_constraint_list_H
#include "Copyright.h"

#include <vector>
#include <memory>

#include "C_production.h"

class LensContext;
class C_separation_constraint;

class C_separation_constraint_list : public C_production
{
   public:
      C_separation_constraint_list(const C_separation_constraint_list&);
      C_separation_constraint_list(C_separation_constraint *, SyntaxError *);
      C_separation_constraint_list(C_separation_constraint_list *, 
				   C_separation_constraint *, SyntaxError *);
      void releaseList(std::unique_ptr<std::vector<C_separation_constraint*> >& 
		       separation_constraints);
      const std::vector<C_separation_constraint*>& getList() const {
	 return *_separation_constraints;
      }
      virtual ~C_separation_constraint_list();
      virtual C_separation_constraint_list* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

   private:
      std::vector<C_separation_constraint*>* _separation_constraints;
};
#endif
