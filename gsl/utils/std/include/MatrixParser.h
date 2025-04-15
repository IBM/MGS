// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef MATRIXPARSER_H
#define MATRIXPARSER_H
#include "Copyright.h"

#include <vector>
#include <string>
#include <list>

class MatrixParser
{
   public:
      enum Type {_TXT, _BMP, _TIFF, _USER_SPECIFIED};
      MatrixParser(Type t, std::string fname);
      MatrixParser(std::vector<int> & size, std::vector<short> & matrix);
      std::vector<short> getMatrix(std::vector<int> & begin, std::vector<int> & end);
      std::vector<short> getMatrix() {return _matrix;}
      std::vector<int>&  getSize() {return _size;}
      MatrixParser::Type getType() {return _type;}
      ~MatrixParser();

   protected:
      void setVector(std::vector<int>&, std::istream&);
      void setVector(std::vector<short>&, std::istream&);

   private:
      Type _type;
      std::vector<short> _matrix;
      std::vector<int> _size;
      std::vector<unsigned> _strides;
      unsigned _dimensions;
};

#endif
