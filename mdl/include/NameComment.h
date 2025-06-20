// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
