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
      void releaseList(std::auto_ptr<std::vector<C_separation_constraint*> >& 
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
