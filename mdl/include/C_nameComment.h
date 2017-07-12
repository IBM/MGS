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

#ifndef C_nameComment_H
#define C_nameComment_H
#include "Mdl.h"

#include "C_production.h"
#include <memory>
#include <string>

class MdlContext;

class C_nameComment : public C_production {

   public:
      virtual void execute(MdlContext* context);
      C_nameComment(); 
      C_nameComment(const std::string& name,
		    const std::string& comment = "", 
		    int blockSize = 0, int incrementSize = 0); 
      C_nameComment(const std::string& name,
		    int blockSize, int incrementSize = 0); 
      C_nameComment(const C_nameComment& rv);
      virtual void duplicate(std::auto_ptr<C_nameComment>& rv) const;
      virtual void duplicate(std::auto_ptr<C_production>& rv) const;
      virtual ~C_nameComment();

      const std::string& getName() const {
	 return _name;
      }

      void setName(const std::string& name)  {
	 _name = name;
      }

      const std::string& getComment() const {
	 return _comment;
      }

      void setComment(const std::string& comment)  {
	 _comment = comment;
      }      
      
      int getBlockSize() const {
	 return _blockSize;
      }

      int getIncrementSize() const {
	 return _incrementSize;
      }

   private:
      std::string _name;
      std::string _comment; 
      int _blockSize;
      int _incrementSize;
};


#endif // C_nameComment_H
