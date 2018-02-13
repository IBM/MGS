#include "../include/FileUtils.h"
#include "../include/StringUtils.h"
#include <string>
#include <sys/stat.h> // hold data return by stat() function
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <sstream>
#include <cassert>


using namespace std;

// Check if the folders in the given path is existed
//   the path may have file name included
//   we use the hints of extension to check
bool FileFolderUtils::isFolderExist(const std::string& path,
                                    const std::string& fileExtensions)
{
  // StringUtils::Tokenize(path, )
  return false;
}

bool FileFolderUtils::isFileExist(const std::string& file_name)
{
  struct stat buffer;
  return (stat (file_name.c_str(), &buffer) == 0);
}

//ASSSUMPTION: # --> comment
// comma or space delimiter
// 1. first non-comment line indicate the space-separated fieldnames  (time, Vm)
// 2. second non-comment line indicate the # of column
std::map< std::string, std::vector<float> > FileFolderUtils::readCsvFile(const std::string& file_name, unsigned int & num_col)
{
  std::ifstream data(file_name);

  //int row_count = 0;
  std::string line;
  //int num_col = 0;
  num_col = 0;
  int index_line = 0;
  //int LINE_START_DATA = 2;
  std::map< std::string, std::vector<float> > results;
  std::vector<std::string>  fieldNames;
  //while(std::getline(data, line))
  for (std::string line; std::getline(data, line); )
  {
    line.erase(line.length()-1);
    if (line.empty() or  line.find_first_of("#") == 0)
      continue; // skip blank line
    index_line += 1;
    if (index_line == 1)
    {
      std::vector<std::string> tokens;
      std::string delimiters=" ,";
      StringUtils::Tokenize(line, tokens, delimiters);
      for (unsigned int i=0; i < tokens.size(); i++)
      {
        results[tokens[i]] = std::vector<float>();
        fieldNames.push_back(tokens[i]);
      }
    }
    else if (index_line == 2)
    {
      std::stringstream lineStream(line);
      lineStream >> num_col;
      assert(num_col == results.size());
    }
    else{
      std::vector<std::string> tokens;
      std::string delimiters=" ,\t";
      StringUtils::Tokenize(line, tokens, delimiters);
      for (unsigned int i=0; i < tokens.size(); i++)
      {
        results[fieldNames[i]].push_back(std::stof(tokens[i]));
      }
    }
  }
  return results;
  //std::vector<std::vector<float> > col_data(num_col, std::vector<float>(1));

  //while(std::getline(data, line))
  //{
  //  if (line.find_first_of("#") == 0)
  //    continue; // skip blank line

  //  row_count += 1;
  //  std::stringstream lineStream(line);
  //  std::string cell;
  //  int column_index = -1;
  //  std::vector<std::string> cells;
  //  std::string delimiter=", ";
  //  StringUtils::Tokenize(line, cells, delimiter);
  //  int num_col = cells.size();
  //  while(std::getline(lineStream, cell, ','))
  //  {
  //    column_index +=1;
  //    //col_data[row_count-1][column_index]
  //  }
  //}
  //return col_data;
}
