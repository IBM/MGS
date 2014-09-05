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

#ifndef C_ARGUMENT_MATRIX_H
#define C_ARGUMENT_MATRIX_H
#include "Copyright.h"

#include <string>
#include <memory>
#include <vector>
#include "C_argument.h"

class C_argument;
class C_type_specifier;
class LensContext;
class DataItem;
class DataItemArrayDataItem;
class IntArrayDataItem;
class FloatArrayDataItem;
class SyntaxError;

class C_argument_matrix: public C_argument
{
   public:
      C_argument_matrix(const C_argument_matrix&);
      C_argument_matrix(C_type_specifier *, C_argument_list *, SyntaxError *);
      virtual ~C_argument_matrix();
      virtual C_argument_matrix* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

      // accessors
      C_argument_list *getArgument_list() { 
	 return _arg_list; 
      }
      DataItem *getArgumentDataItem() const;
      C_type_specifier *getTypeSpecifier() const { 
	 return _typeSpec; 
      }

   private:
      DataItemArrayDataItem* _di_array_di;
      IntArrayDataItem* _int_array_di;
      FloatArrayDataItem* _float_array_di;
      C_argument_list* _arg_list;
      C_type_specifier* _typeSpec;
};
#endif
