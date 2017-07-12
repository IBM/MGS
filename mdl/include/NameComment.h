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

#ifndef NameComment_H
#define NameComment_H
#include "Mdl.h"

#include <string>


class NameComment {

   public:
      NameComment(const std::string& name = "",
		  const std::string& comment = "",
		  int blockSize = 0,
		  int incrementSize = 0);
      
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

      ~NameComment();
   private:
      std::string _name;
      std::string _comment; 
      int _blockSize;
      int _incrementSize;
};


#endif // C_nameComment_H
