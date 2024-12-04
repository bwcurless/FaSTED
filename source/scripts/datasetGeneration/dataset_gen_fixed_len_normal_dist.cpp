//We make normally (gaussian) distributed datasets
//Needed so that we get more neighbors in comparison to uniformly distributed data
//make in the range 0-1, so we don't need to re-normalize for Super-EGO

//Makefile:
//g++ -std=c++11 -O3 dataset_gen_fixed_len_normal_dist.cpp -o dataset_gen_fixed_len_normal_dist
//

#include <stdio.h>
#include <random>
#include <fstream>
#include <math.h>
#include <iostream>
#include <string.h>
#include <iostream>

//static seed so we can reproduce the data on other machines
#define SEED 2137834274 


using namespace std;

int main(int argc, char *argv[])
{


if (argc!=3)
{
cout <<"\n\nIncorrect number of input parameters.  \nShould be: num. dimensions data points\n E.g., ./dataset_gen_fixed_len_normal_dist 2 2000000\n";
return 0;
}

	char inputnumdim[500];
	char inputdatapoints[500];

	strcpy(inputnumdim,argv[1]);
	strcpy(inputdatapoints,argv[2]);
	
unsigned int numDim=atoi(inputnumdim);
unsigned int dataPoints=atoi(inputdatapoints);


unsigned int length=1;
// unsigned int numDim=8;
// unsigned int dataPoints=100000000;


printf("\nTotal datapoints: %d",dataPoints);
double datasetsize=((dataPoints*8.0*numDim)/(1024.0*1024.0));
printf("Size of dataset (MiB): %f",datasetsize);


std::ofstream myfile;
std::string fname="dataset_fixed_len_pts_NDIM_";
fname+=std::to_string(numDim);
fname+="_pts_"; 
fname+=std::to_string(dataPoints);
fname+=".txt"; 
myfile.open(fname);






// std::random_device rd;  //Will be used to obtain a seed for the random number engine
std::mt19937 gen(SEED); //Standard mersenne_twister_engine seeded with rd()
std::normal_distribution<double> dis(0.5,0.05); //mean, stardard deviation

double total=0;

for (int i=0; i<dataPoints; i++){
	for (int j=0; j<numDim; j++){
		double val=0;
		//generate value until its in the range 0-1
		do {
		val=dis(gen)*length;
		}while (val<0 || val>1);

		total+=val;	
		
		if (j<numDim-1){
			myfile<<val<<", ";
		}
		else{
			myfile<<val;	
		}
	}
	myfile<<std::endl;
}

printf("\nAverage of values generated: %f",total/(dataPoints*numDim*1.0));

myfile.close();

printf("\n\n");

return 0;
}