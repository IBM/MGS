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
