#ifndef __FILEFOLDERUTILS_H__
#define __FILEFOLDERUTILS_H__

#include <string>
#include <vector>
#include <map>

class FileFolderUtils
{
  public:
  static bool isFolderExist(const std::string& str,
                            const std::string& fileExtension = ".dat,.txt");
  static bool isFileExist(const std::string& file_name);

  static std::map< std::string, std::vector<float> > readCsvFile(const std::string& file_name, unsigned int & num_col);
};

#endif
