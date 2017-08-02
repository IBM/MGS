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

#ifndef C_non_initializable_type_specifier_H
#define C_non_initializable_type_specifier_H
#include "Copyright.h"

#include "C_type_specifier.h"
#include "C_production.h"

class LensContext;
class SyntaxError;

class C_non_initializable_type_specifier : public C_production
{
   public:
      C_non_initializable_type_specifier(
	 const C_non_initializable_type_specifier&);
      C_non_initializable_type_specifier(
	 C_type_specifier::Type, SyntaxError *);
      virtual ~C_non_initializable_type_specifier ();
      virtual C_non_initializable_type_specifier* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      C_type_specifier::Type getType() const { 
	 return _type;
      }
      C_type_specifier* getNextTypeSpecifier() {
	 return _nextTypeSpec;
      }

   private:
      C_type_specifier::Type _type;
      C_type_specifier* _nextTypeSpec;
};
#endif
