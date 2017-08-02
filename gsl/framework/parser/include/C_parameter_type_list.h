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
