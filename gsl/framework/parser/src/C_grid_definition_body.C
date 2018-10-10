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

#include "C_grid_definition_body.h"
#include "C_int_constant_list.h"
#include "C_dim_declaration.h"
#include "C_grid_translation_unit.h"
#include "Repertoire.h"
#include "RepertoireDataItem.h"
#include "Grid.h"
#include "LensContext.h"
#include "SyntaxError.h"
#include "SyntaxErrorException.h"
#include "C_production.h"

void C_grid_definition_body::internalExecute(LensContext *c)
{
   _dimDecl->execute(c);
   // execute is not called on _gridTransUnit until we actually try to create 
   // a gridRepertoire in createRepertoire() method
}


C_grid_definition_body::C_grid_definition_body(
   const C_grid_definition_body& rv)
   : C_production(rv), _dimDecl(0), _gridTransUnit(0)
{
   if (rv._dimDecl) {
      _dimDecl = rv._dimDecl->duplicate();
   }
   if (rv._gridTransUnit) {
      _gridTransUnit = rv._gridTransUnit->duplicate();
   }
}


C_grid_definition_body::C_grid_definition_body(
   C_dim_declaration *d, SyntaxError * error)
   : C_production(error), _dimDecl(d), _gridTransUnit(0)
{
}


C_grid_definition_body::C_grid_definition_body(
   C_dim_declaration *d, C_grid_translation_unit *g, SyntaxError * error)
   : C_production(error), _dimDecl(d), _gridTransUnit(g)
{
}


C_grid_definition_body* C_grid_definition_body::duplicate() const
{
   return new C_grid_definition_body(*this);
}

void C_grid_definition_body::duplicate(
   std::unique_ptr<RepertoireFactory>& rv) const
{
   rv.reset(new C_grid_definition_body(*this));
}

C_grid_definition_body::~C_grid_definition_body()
{
   delete _dimDecl;
   delete _gridTransUnit;
}


Repertoire* C_grid_definition_body::createRepertoire(
   std::string const& repName, LensContext* c)
{
   std::vector<int> size;
   // _dimDecl is dim declaration
   // _icl is int constant list
   std::list<int> const * lint = _dimDecl->getIntConstantList()->getList();
   std::list<int>::const_reverse_iterator int_iter, int_end = lint->rend();
   for(int_iter = lint->rbegin(); int_iter != int_end; ++int_iter) {
      size.push_back(*int_iter);
   }

   Repertoire* gridRep = new Repertoire(repName);
   Grid* grid = gridRep->setGrid(size);

   std::string currentRep("CurrentRepertoire");
   const RepertoireDataItem* crdi = dynamic_cast<const RepertoireDataItem*>(
      c->symTable.getEntry(currentRep));
   if (crdi == 0) {
      std::string mes = 
	 "dynamic cast of DataItem to RepertoireDataItem failed";
      throwError(mes);
   }
   Repertoire* parentRep = crdi->getRepertoire();

   std::unique_ptr<Repertoire> rap(gridRep);
   parentRep->addSubRepertoire(rap);
   //gridRep->setParentRepertoire(parentRep);

   // put new repertoire in symbol table in the current scope
    RepertoireDataItem * rdi = new RepertoireDataItem;
    rdi->setRepertoire(gridRep);
    std::unique_ptr<DataItem> di_ap(rdi);

   try {
      c->symTable.addEntry(repName, di_ap);
   } catch (SyntaxErrorException& e) {
      throwError("While adding grid definition body " + 
		 repName + " " + e.getError());
   }
   c->symTable.addLocalScope();
   try {
      c->addCurrentRepertoire(gridRep);
   } catch (SyntaxErrorException& e) {
      throwError("While adding grid definition body " + 
		 currentRep + " " + e.getError());
   }
   
   // Create a copy so that the original does not get modified.
   C_grid_translation_unit* localCopy = 
      new C_grid_translation_unit(*_gridTransUnit);
   try {
      localCopy->execute(c, grid);
   } catch (SyntaxErrorException& e) {
      e.printError();
      e.resetError();
      setError();
      localCopy->recursivePrint();
      localCopy->printTdError();
      throw;
   }
   c->symTable.removeLocalScope();
   delete localCopy;

   return gridRep;
}

void C_grid_definition_body::checkChildren() 
{
   if (_dimDecl) {
      _dimDecl->checkChildren();
      if (_dimDecl->isError()) {
         setError();
      }
   }
   if (_gridTransUnit) {
      _gridTransUnit->checkChildren();
      if (_gridTransUnit->isError()) {
         setError();
      }
   }
} 

void C_grid_definition_body::recursivePrint() 
{
   if (_dimDecl) {
      _dimDecl->recursivePrint();
   }
   if (_gridTransUnit) {
      _gridTransUnit->recursivePrint();
   }
   printErrorMessage();
} 


void C_grid_definition_body::setTdError(SyntaxError *tdError)
{
   if (_gridTransUnit) {
      _gridTransUnit->setTdError(tdError);
   }
}
