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

#ifndef C_ARGUMENT_ARGUMENT_LIST_H
#define C_ARGUMENT_ARGUMENT_LIST_H
#include "Copyright.h"

#include <string>
#include <memory>
#include <vector>
#include "C_argument.h"

class C_argument;
class C_argument_list;
class C_type_specifier;
class LensContext;
class DataItem;
class DataItemArrayDataItem;
class IntArrayDataItem;
class FloatArrayDataItem;
class SyntaxError;

class C_argument_argument_list: public C_argument
{
   public:
      C_argument_argument_list(const C_argument_argument_list&);
      C_argument_argument_list(C_argument_list *, SyntaxError *);
      C_argument_argument_list(C_type_specifier *, C_argument_list *, 
			       SyntaxError *);
      virtual ~C_argument_argument_list();
      virtual C_argument_argument_list* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

      // accessors
      C_argument_list *getArgument_list() { 
	 return _arg_list; 
      }
      DataItem *getArgumentDataItem() const {
	 return _dataItem;
      }
      C_type_specifier *getTypeSpecifier() const { 
	 return _typeSpec; 
      }

   private:
      DataItem* _dataItem;
      C_argument_list* _arg_list;
      C_type_specifier* _typeSpec;
};
#endif
