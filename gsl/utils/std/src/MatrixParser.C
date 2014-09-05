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

#ifdef HAVE_MPI
#include <mpi.h>
#endif
#include "MatrixParser.h"
#include "VolumeOdometer.h"
#include "VectorOstream.h"
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

MatrixParser::MatrixParser(Type t, std::string fname)
: _type(t)
{
   if (_type == _TXT) {
      std::ifstream infile(fname.c_str());
      if (!infile.is_open()) {
         std::cerr<<"Failed to open specified txt file "<<fname<<" in MatrixParser!"<<std::endl;
         exit(-1);
      }
      infile>>_dimensions;
      unsigned sz = 1;
      for (unsigned i = 0; i<_dimensions; ++i) {
         int dim;
         infile>>dim;
         _size.push_back(dim);
         _strides.push_back(sz);
         sz *= dim;
      }
      std::vector<short> in_buffer;
      in_buffer.resize(sz);
      for (unsigned i = 0; i<sz && !infile.eof(); ++i) infile>>in_buffer[i];
      infile.close();

      if (_dimensions == 1) {
         for (unsigned i = 0; i<sz; ++i) _matrix.push_back(in_buffer[i]);
      }
      else if (_dimensions == 2) {
         int ii, jj;
         int nrows = _size[0];
         int ncols = _size[1];
         for(jj = 0; jj < ncols; ++jj) {
            for(ii = nrows-1; ii >=0; --ii) {
               _matrix.push_back(in_buffer[(ii*ncols)+jj]);
            }
         }
      }
      else if (_dimensions == 3) {
         int ii, jj, kk;
         int nrows = _size[0];
         int ncols = _size[1];
         int nslices = _size[2];
         for(kk = 0; kk < nslices; ++kk) {
            int sliceStride = kk*nrows*ncols;
            for(jj = 0; jj < ncols; ++jj) {
               for(ii = nrows-1; ii >=0; --ii) {
                  _matrix.push_back(in_buffer[sliceStride+(ii*ncols)+jj]);
               }
            }
         }
      }
      else {
         std::cerr<<"MatrixParser : Text matrices of only 1, 2, and 3 dimensions supported!"<<std::endl;
         exit(-1);
      }
      if (_matrix.size() != sz) {
         std::cerr<<"Specified matrix and dimensions don't match!"<<std::endl;
         exit(-1);
      }
   }
   else if (_type == _BMP) {
      std::cerr<<"Specified type "<<_type<<" not implemented on MatrixParser!"<<std::endl;
      exit(-1);
   }
   else if (_type == _TIFF) {
      int nrows, ncols;
      unsigned char *in_buffer;
      int ReadImage(char *, unsigned char **, int *, int *);
      char filename[100];
      strcpy(filename, fname.c_str());
      if(! ReadImage(filename, &in_buffer, &nrows, &ncols)) {
         std::cerr << "Problem reading input file " << filename[1] << std::endl;
      }
      _dimensions = 2;
      _size.push_back(nrows);
      _size.push_back(ncols);
      _strides.push_back(1);
      _strides.push_back(nrows);
      int ii, jj;
      for(jj = 0; jj < ncols; jj++) {
         for(ii = nrows-1; ii >=0; ii--) {
            _matrix.push_back(in_buffer[ii*ncols+jj]);
         }
      }
   }
}


MatrixParser::MatrixParser(std::vector<int> & size, std::vector<short> & m)
: _type(_USER_SPECIFIED), _matrix(m), _size(size)
{
   _dimensions = _size.size();
   unsigned sz = 1;
   for (unsigned i = 0; i<_dimensions; ++i) {
      _strides[i] = sz;
      sz *= _size[i];
   }
   if (_matrix.size() != sz) {
      std::cerr<<"Specified matrix and dimensions don't match!"<<std::endl;
      exit(-1);
   }
}


std::vector<short> MatrixParser::getMatrix(std::vector<int> & begin, std::vector<int> & end)
// This method assumes coordinates begin and end are ordered from most significant to least
// significant, as from an odometer
{
   std::vector<short> rval;
   if ( (_dimensions != begin.size()) || (_dimensions != end.size()) ) {
      std::cerr<<"Requested matrix and MatrixParser are of different dimensions : "<<_dimensions<<" != "
         <<begin.size()<<"/"<<end.size()<<"!"<<std::endl;
      exit(-1);
   }

   for (unsigned i = 0; i<_dimensions; ++i) {
      if (end[i] > _size[i]) end[i]=_size[i];
      if (end[i] < begin[i]) {
         std::cerr<<"Begin and end coordinates are reversed in MatrixParser!"<<std::endl;
         exit(-1);
      }
   }
   VolumeOdometer odmtr(begin,end);
   std::vector<unsigned>::iterator stridesEnd =_strides.end();
   std::vector<int> & coords = odmtr.look();
   for (; !odmtr.isRolledOver(); odmtr.next() ) {
      std::vector<int>::const_iterator ci = coords.begin();
      std::vector<unsigned>::iterator si = _strides.begin();
      unsigned index=0;
      do {
         index += *si++ * *ci++;
      } while(si!=stridesEnd);
      rval.push_back(_matrix[index]);
   }
   return rval;
}


MatrixParser::~MatrixParser()
{
}

void MatrixParser::setVector(std::vector<int>& v, std::istream& is)
{
   int i,j,k;
   v.clear();
   is>>i;
   for (j=0;j<i;++j) {
      is>>k;
      v.push_back(k);
   }
}


void MatrixParser::setVector(std::vector<short>& v, std::istream& is)
{
   int i,j;
   short k;
   v.clear();
   is>>i;
   for (j=0;j<i;++j) {
      is>>k;
      v.push_back(k);
   }
}
