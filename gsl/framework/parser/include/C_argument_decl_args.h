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

#ifndef C_ARGUMENT_DECL_ARGS_H
#define C_ARGUMENT_DECL_ARGS_H
#include "Copyright.h"

#include <string>
#include <memory>
#include "C_argument.h"

class LensContext;
class DataItem;
class C_declarator;
class C_argument_list;
class SyntaxError;

class C_argument_decl_args: public C_argument
{
   public:
      C_argument_decl_args(const C_argument_decl_args&);
      C_argument_decl_args(C_declarator *dcl, C_argument_list *, 
			   SyntaxError *);
      virtual ~C_argument_decl_args();
      virtual C_argument_decl_args* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

      // accessors
      DataItem *getArgumentDataItem() const {
	 return _dataItem;
      }

   private:
      C_declarator* _declarator;
      C_argument_list* _argList;
      DataItem* _dataItem;
};
#endif
