#include "Mgs.h"
#include "LoadMatrix.h"
#include "CG_LoadMatrixBase.h"
#include "LensContext.h"
#include "ShallowArray.h"
#include <memory>
#include <fstream>

void LoadMatrix::userInitialize(LensContext* CG_c, CustomString& filename, int& rows, int& cols) 
{
}

ShallowArray< float > LoadMatrix::userExecute(LensContext* CG_c) 
{
  ShallowArray< float > rval;

  std::ifstream input;
  input.open (init.filename.c_str(), std::ios::binary );

  if (!input.is_open()) {
    std::cerr<<"LoadMatrix : Error opening file : "<<init.filename<<"."<<std::endl;
    exit(0);
  }

  int fsz = 0;
  input.seekg (0, std::ios::end);
  fsz = input.tellg();
  input.seekg (0, std::ios::beg);
  long csz = sizeof(float)*init.rows;

  if (fsz != init.cols * csz) {
    std::cerr<<"LoadMatrix : "<<init.filename<<" file size ("<<fsz<<") does not match arguments. "<<init.cols*csz<<" expected."<<std::endl;
    exit(0);
  }

  rval.increaseSizeTo(init.rows*init.cols);
  ShallowArray< float >::iterator iter = rval.begin();
  float* buffer = new float[init.rows];

  for (int i=0; i<init.cols; ++i) {    
    input.read(reinterpret_cast<char*>(buffer),csz);
    if (!input.good()) {
      std::cerr<<"LoadMatrix : Error reading file : "<<init.filename<<"."<<std::endl;
      exit(0);
    }
    for (int j=0; j<init.rows; ++j, ++iter)
      (*iter)=buffer[j];
  }
  input.close();
  delete [] buffer;
  return rval;
}

LoadMatrix::LoadMatrix() 
   : CG_LoadMatrixBase()
{
}

LoadMatrix::~LoadMatrix() 
{
}

void LoadMatrix::duplicate(std::unique_ptr<LoadMatrix>&& dup) const
{
   dup.reset(new LoadMatrix(*this));
}

void LoadMatrix::duplicate(std::unique_ptr<Functor>&& dup) const
{
   dup.reset(new LoadMatrix(*this));
}

void LoadMatrix::duplicate(std::unique_ptr<CG_LoadMatrixBase>&& dup) const
{
   dup.reset(new LoadMatrix(*this));
}

