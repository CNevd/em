#include <fstream>   

int main () {

  std::ofstream outfile ("data.txt",std::ofstream::binary);

  int dim = 3,size = 10;
  outfile.write ((char*)(&size),sizeof(int));
  outfile.write ((char*)(&dim),sizeof(int));

  double data[10][3] = {
        0.0, 0.2, 0.4,
        0.3, 0.2, 0.4,
        0.4, 0.2, 0.4,
        0.5, 0.2, 0.4,
        5.0, 5.2, 8.4,
        6.0, 5.2, 7.4,
        4.0, 5.2, 4.4,
        10.3, 10.4, 10.5,
        10.1, 10.6, 10.7,
        11.3, 10.2, 10.9
    };

  for(int i = 0; i < 10; i++){
    for(int j = 0; j < 3; j++)
	outfile.write ((char*)(&(data[i][j])),sizeof(double));
  }
  outfile.close();

  return 0;
}
