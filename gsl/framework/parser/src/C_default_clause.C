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

#include "C_default_clause.h"
#include "C_constant.h"
#include "IntArrayDataItem.h"
#include "FloatArrayDataItem.h"
#include "SyntaxError.h"
#include "SyntaxErrorException.h"
#include "C_production_adi.h"


void C_default_clause::internalExecute(LensContext *c, ArrayDataItem *adi)
{

   _constant->execute(c);

   if (_constant->getType() == C_constant::_INT) {
      int I_defaultConst = _constant->getInt();

      // need to do a dynamic cast to intarray or float array data item ??

      if (IntArrayDataItem *I_adi = dynamic_cast<IntArrayDataItem*>(adi))
         fill (I_adi->getModifiableIntVector()->begin(), 
	       I_adi->getModifiableIntVector()->end(), I_defaultConst );
      else {
         FloatArrayDataItem *F_adi = dynamic_cast<FloatArrayDataItem*>(adi);
         fill (F_adi->getModifiableFloatVector()->begin(), 
	       F_adi->getModifiableFloatVector()->end(), 
	       (float) I_defaultConst);
      }
   }
   else if( _constant->getType() == C_constant::_FLOAT) {
      float F_defaultConst = _constant->getFloat();
      if (IntArrayDataItem *I_adi = dynamic_cast<IntArrayDataItem*>(adi))
         fill (I_adi->getModifiableIntVector()->begin(), 
	       I_adi->getModifiableIntVector()->end(), (int) F_defaultConst);
      else {
         FloatArrayDataItem *F_adi = dynamic_cast<FloatArrayDataItem*>(adi);
         fill (F_adi->getModifiableFloatVector()->begin(), 
	       F_adi->getModifiableFloatVector()->end(), F_defaultConst);
      }
   }
   else {
      std::string mes = "wrongly defined default constant type";
      throwError(mes);
   }

}


C_default_clause::C_default_clause(const C_default_clause& rv)
   : C_production_adi(rv), _constant(0)
{
   if (rv._constant) {
      _constant = rv._constant->duplicate();
   }
}


C_default_clause::C_default_clause(C_constant *c, SyntaxError * error)
   : C_production_adi(error), _constant(c)
{
}


C_default_clause* C_default_clause::duplicate() const
{
   return new C_default_clause(*this);
}


C_default_clause::~C_default_clause()
{

   delete _constant;

}

void C_default_clause::checkChildren() 
{
   if (_constant) {
      _constant->checkChildren();
      if (_constant->isError()) {
         setError();
      }
   }
} 

void C_default_clause::recursivePrint() 
{
   if (_constant) {
      _constant->recursivePrint();
   }
   printErrorMessage();
} 
