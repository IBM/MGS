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

#include "C_edgeConnection.h"
#include "C_connection.h"
#include "C_general.h"
#include "C_generalList.h"
#include "MdlContext.h"
#include "C_interfacePointerList.h"
#include "Edge.h"
#include "Connection.h"
#include "EdgeConnection.h"
#include <memory>
#include <string>
#include <iostream>

void C_edgeConnection::execute(MdlContext* context, 
			       Edge* edge) 
{
   EdgeConnection* edgeConnection;

   edgeConnection = new EdgeConnection(_type);

   doConnectionWork(context, edge, edgeConnection);
   
   std::unique_ptr<EdgeConnection> aConnection;
   aConnection.reset(edgeConnection);
   
   switch (_type) {
   case Connection::_PRE:
      edge->setPreNode(std::move(aConnection));
      break;
   case Connection::_POST:
      edge->setPostNode(std::move(aConnection));
      break;
   }
}

C_edgeConnection::C_edgeConnection() 
   : C_connection(), _type(Connection::_PRE)
{
}

C_edgeConnection::C_edgeConnection(
   C_interfacePointerList* ipl, C_generalList* gl, 
   Connection::DirectionType type)
   : C_connection(ipl, gl), _type(type)
{
}

void C_edgeConnection::duplicate(std::unique_ptr<C_edgeConnection>&& rv) const
{
   rv.reset(new C_edgeConnection(*this));
}

void C_edgeConnection::duplicate(std::unique_ptr<C_connection>&& rv) const
{
   rv.reset(new C_edgeConnection(*this));
}

void C_edgeConnection::duplicate(std::unique_ptr<C_general>&& rv) const
{
   rv.reset(new C_edgeConnection(*this));
}

C_edgeConnection::~C_edgeConnection() 
{
}

void C_edgeConnection::addToList(C_generalList* gl) 
{
   std::unique_ptr<C_edgeConnection> con;
   con.reset(new C_edgeConnection(*this));
   switch (_type) {
   case Connection::_PRE:
      gl->addPreNode(std::move(con));
      break;
   case Connection::_POST:
      gl->addPostNode(std::move(con));
      break;
   }
}

std::string C_edgeConnection::getTypeStr() const
{
   std::string retVal = "not set";
   switch (_type) {
   case Connection::_PRE:
      retVal = "PreNode";
      break;
   case Connection::_POST:
      retVal = "PostNode";
      break;
   }
   return retVal;
}
