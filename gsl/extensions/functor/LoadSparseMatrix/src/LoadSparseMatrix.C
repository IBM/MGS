#include "Lens.h"
#include "LoadSparseMatrix.h"
#include "CG_LoadSparseMatrixBase.h"
#include "LensContext.h"
#include "ShallowArray.h"
#include <memory>
#include <fstream>

void LoadSparseMatrix::userInitialize(LensContext* CG_c, String& filename) 
{
}

ShallowArray<float> LoadSparseMatrix::userExecute(LensContext* CG_c) 
{
  std::ifstream input;
  std::ostringstream os;
  os.str("");
  os << init.filename; 
  input.open(os.str().c_str(), std::ifstream::app|std::ifstream::in);

  if (!input.is_open()) {
    std::cerr<<"LoadSparseMatrix : Error opening file : "<< init.filename <<"."<<std::endl;
    exit(0);
  }
  ShallowArray<float> rval; 
  float row, col, val;
  for(int i=0; !input.eof(); i++) {
    input >> row >> col >> val;
    rval.increaseSizeTo(rval.size()+3);
    rval[i*3] = row;
    rval[i*3+1] = col;
    rval[i*3+2] = val;
  }
  input.close();
  std::cout<<rval<<std::endl;
  return rval;
}

LoadSparseMatrix::LoadSparseMatrix() 
   : CG_LoadSparseMatrixBase()
{
}

LoadSparseMatrix::~LoadSparseMatrix() 
{
}

void LoadSparseMatrix::duplicate(std::auto_ptr<LoadSparseMatrix>& dup) const
{
   dup.reset(new LoadSparseMatrix(*this));
}

void LoadSparseMatrix::duplicate(std::auto_ptr<Functor>& dup) const
{
   dup.reset(new LoadSparseMatrix(*this));
}

void LoadSparseMatrix::duplicate(std::auto_ptr<CG_LoadSparseMatrixBase>& dup) const
{
   dup.reset(new LoadSparseMatrix(*this));
}

