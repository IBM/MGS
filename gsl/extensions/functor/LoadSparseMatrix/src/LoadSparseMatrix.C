#include "Lens.h"
#include "LoadSparseMatrix.h"
#include "CG_LoadSparseMatrixBase.h"
#include "LensContext.h"
#include "ShallowArray.h"
#include <memory>
#include <fstream>

//#edefine DBG

void LoadSparseMatrix::userInitialize(LensContext* CG_c, String& filepath, String& filename) 
{
}

ShallowArray<float> LoadSparseMatrix::userExecute(LensContext* CG_c) 
{
  std::ifstream input;
  std::ostringstream os;
  os.str("");
  os << (init.filepath + init.filename); 
  input.open(os.str().c_str(), std::ifstream::in); //std::ifstream::app|std::ifstream::in);

  if (!input.is_open()) {
    std::cerr<<"LoadSparseMatrix : Error opening file : "<< init.filepath + init.filename <<"."<<std::endl;
    exit(0);
  }

//  int fsz = 0;
//  input.seekg (0, std::ios::end);
//  fsz = input.tellg();
//  input.seekg (0, std::ios::beg);
//  long csz = sizeof(float)*init.rows;

//  if (fsz != init.cols * csz) {
//    std::cerr<<"LoadSparseMatrix : "<<init.filename<<" file size ("<<fsz<<") does not match arguments. "<<init.cols*csz<<" expected."<<std::endl;
//    exit(0);
//  }

  std::string line;
  int nb_lines = 0; 
  while(std::getline(input, line)) {
    nb_lines++;
  }

  input.clear();
  input.seekg(0, std::ios::beg);

  ShallowArray<float> rvals;
  rvals.increaseSizeTo(nb_lines*3); 
  int row, col;
  float val;

  ShallowArray<float>::iterator it = rvals.begin();
  while(std::getline(input, line)) {
    //input >> row >> col >> val;
    //std::cout << "input.eof(): " << input.eof() << std::endl << std::endl;
    //std::cout << "rvals.size() = " << rvals.size() << std::endl;
    //rvals.increaseSizeTo(sizeof(float)*(rvals.size()+3));
    //std::getline(input, line);
    std::istringstream iss(line);
    iss >> row >> col >> val;
    //std::cout << line << std::endl;
    std::cout << "row = " << row << ";  col = " << col << ";  val = " << val << std::endl;
    //rval.increaseSizeTo(rval.size()+3);
    *it = static_cast<float>(row); it++;
    *it = static_cast<float>(col); it++;
    *it = static_cast<float>(val); it++;
    //std::cout << *it << " " << *(it+1) << " " << *(it+2) << std::endl;
  }
  input.close();
  
  /*
  it = rvals.begin();
  ShallowArray<float>::iterator end = rvals.end();
  for (it; it!= end; it+=3){
    std::cout << static_cast<int>(*it) << " " << static_cast<int>(*(it+1)) << " " << static_cast<float>(*(it+2)) << std::endl;
  }
  */
  //std::cout<< rval <<std::endl;
  return rvals;
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

