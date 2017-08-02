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

#include "C_declaration_repertoire.h"
#include "C_repertoire_declaration.h"
#include "LensContext.h"
#include "RepertoireFactory.h"
#include "RepertoireFactoryDataItem.h"
#include "SyntaxError.h"
#include "SyntaxErrorException.h"


void C_declaration_repertoire::internalExecute(LensContext *c)
{
   _repertoireDeclaration->execute(c);

   // We don't allow forward declarations, hence its type must be 
   // in symbol table.

   std::string type = _repertoireDeclaration->getType();
   std::string instance = _repertoireDeclaration->getName();

   const DataItem* di = c->symTable.getEntry(type);
   if(di == 0) {
      // Forward declaration, throw an error and maybe let it 
      // parse the rest of the spec file, and then exit.
      std::string mes = " symbol " + type + " not found";
      throwError(mes);
   }
   const RepertoireFactoryDataItem *rdi = 
      dynamic_cast<const RepertoireFactoryDataItem*>(di);
   if (rdi != 0)
      rdi->getFactory()->createRepertoire(instance, c);
   else {
      // Throw error, exit after parsing.
      std::string mes = "dynamic cast of symbol " + 
	 type + "'s DataItem to RepertoireFactoryDataItem failed";
      throwError(mes);
   }
}


C_declaration_repertoire* C_declaration_repertoire::duplicate() const
{
   return new C_declaration_repertoire(*this);
}


C_declaration_repertoire::C_declaration_repertoire(
   C_repertoire_declaration *r, SyntaxError * error)
   : C_declaration(error), _repertoireDeclaration(r)
{
}


C_declaration_repertoire::C_declaration_repertoire(
   const C_declaration_repertoire& rv)
   : C_declaration(rv), _repertoireDeclaration(0)
{
   if (rv._repertoireDeclaration) {
      _repertoireDeclaration = rv._repertoireDeclaration->duplicate();
   }
}


C_declaration_repertoire::~C_declaration_repertoire()
{
   delete _repertoireDeclaration;
}

void C_declaration_repertoire::checkChildren() 
{
   if (_repertoireDeclaration) {
      _repertoireDeclaration->checkChildren();
      if (_repertoireDeclaration->isError()) {
         setError();
      }
   }
} 

void C_declaration_repertoire::recursivePrint() 
{
   if (_repertoireDeclaration) {
      _repertoireDeclaration->recursivePrint();
   }
   printErrorMessage();
} 
