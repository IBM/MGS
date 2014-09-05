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

#ifndef C_edgeConnection_H
#define C_edgeConnection_H
#include "Mdl.h"

#include "C_connection.h"
#include "Connection.h"
#include <memory>
#include <string>

class MdlContext;
class C_generalList;
class C_interfacePointerList;
class Edge;

class C_edgeConnection : public C_connection {

   public:
      virtual void execute(MdlContext* context,
			   Edge* edge);
      virtual void addToList(C_generalList* gl);
      C_edgeConnection();
      C_edgeConnection(C_interfacePointerList* ipl, C_generalList* gl,
		   Connection::DirectionType type);
      virtual void duplicate(std::auto_ptr<C_edgeConnection>& rv) const;
      virtual void duplicate(std::auto_ptr<C_connection>& rv) const;
      virtual void duplicate(std::auto_ptr<C_general>& rv) const;
      virtual ~C_edgeConnection();

   protected:
      virtual std::string getTypeStr() const;
     
   private:
      Connection::DirectionType _type;
};


#endif // C_edgeConnection_H
