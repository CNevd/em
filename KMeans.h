#ifndef KMEANS_H_
#define KMEANS_H_

#include <fstream>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <iostream>
#include <assert.h>
#include <string.h>
#include "EM.h"

using namespace std;

namespace em{

class KMeans : public EM{
public:
	enum InitMode
	{
		InitRandom,
		InitManual,
		InitUniform,
	};
	enum Distance
	{
		Euclidean,
	};


	KMeans(const int dimNum,const int clusterNum) : EM(dimNum,clusterNum)
	{
		m_dimNum = dimNum;
		m_cateNum = clusterNum;

		m_means = new double*[m_cateNum];
		for (int i = 0; i < m_cateNum; i++)
		{
			m_means[i] = new double[m_dimNum];
			memset(m_means[i], 0, sizeof(double) * m_dimNum);
		}

		m_initMode = InitRandom;
		m_Distance = Euclidean;
		m_maxIter = 100;
		m_criteError = 0.001;
	}

	~KMeans(){
		for (int i = 0; i < m_cateNum; i++)
		{
			delete[] m_means[i];
		}
		delete[] m_means;
	}

	void SetMean(int i, const double* u){ memcpy(m_means[i], u, sizeof(double) * m_dimNum); }
	void SetInitMode(int i)				{ m_initMode = i; }

	double* GetMean(int i)	{ return m_means[i]; }
	int GetInitMode()		{ return m_initMode; }
	


	/*	SampleFile: <size><dim><data>...
		LabelFile:	<size><label>...
	*/
	void Cluster(const char* sampleFileName, const char* labelFileName);
	void Init(std::ifstream& sampleFile);
	void Init(double *data, int N);
	void Cluster(double *data, int N, int *Label);
	friend std::ostream& operator<<(std::ostream& out, KMeans& kmeans);

private:
	double** m_means;
	int m_initMode;
	int m_Distance;
	double GetLabel(const double* x, int* label);
	double CalcDistance(const double* x, const double* u, int dimNum);
};

void KMeans::Init(double *data, int N)
{
	int size = N;
	if (m_initMode ==  InitRandom)
	{
		int inteval = size / m_cateNum;
		double* sample = new double[m_dimNum];

		// Seed the random-number generator with current time
		srand((unsigned)time(NULL));

		for (int i = 0; i < m_cateNum; i++)
		{
			int select = inteval * i + (inteval - 1) * rand() / RAND_MAX;
			for(int j = 0; j < m_dimNum; j++)
				sample[j] = data[select*m_dimNum+j];
			memcpy(m_means[i], sample, sizeof(double) * m_dimNum);
		}

		delete[] sample;
	}
	else if (m_initMode == InitUniform)
	{
		double* sample = new double[m_dimNum];

		for (int i = 0; i < m_cateNum; i++)
		{
			int select = i * size / m_cateNum;
			for(int j = 0; j < m_dimNum; j++)
				sample[j] = data[select*m_dimNum+j];
			memcpy(m_means[i], sample, sizeof(double) * m_dimNum);
		}

		delete[] sample;
	}
	else if (m_initMode == InitManual)
	{
		// Do nothing
	}
}

void KMeans::Init(ifstream& sampleFile)
{
	int size = 0;
	sampleFile.seekg(0, ios_base::beg);
	sampleFile.read((char*)&size, sizeof(int));

	if (m_initMode ==  InitRandom)
	{
		int inteval = size / m_cateNum;
		double* sample = new double[m_dimNum];

		// Seed the random-number generator with current time
		srand((unsigned)time(NULL));

		for (int i = 0; i < m_cateNum; i++)
		{
			int select = inteval * i + (inteval - 1) * rand() / RAND_MAX;
			int offset = sizeof(int) * 2 + select * sizeof(double) * m_dimNum;

			sampleFile.seekg(offset, ios_base::beg);
			sampleFile.read((char*)sample, sizeof(double) * m_dimNum);
			memcpy(m_means[i], sample, sizeof(double) * m_dimNum);
		}

		delete[] sample;
	}
	else if (m_initMode == InitUniform)
	{
		double* sample = new double[m_dimNum];

		for (int i = 0; i < m_cateNum; i++)
		{
			int select = i * size / m_cateNum;
			int offset = sizeof(int) * 2 + select * sizeof(double) * m_dimNum;

			sampleFile.seekg(offset, ios_base::beg);
			sampleFile.read((char*)sample, sizeof(double) * m_dimNum);
			memcpy(m_means[i], sample, sizeof(double) * m_dimNum);
		}

		delete[] sample;
	}
	else if (m_initMode == InitManual)
	{
		// Do nothing
	}
}

double KMeans::CalcDistance(const double* x, const double* u, int dimNum)
{
	double temp = 0;
	if (m_Distance == Euclidean)
	{
		for (int d = 0; d < dimNum; d++)
		{
			temp += (x[d] - u[d]) * (x[d] - u[d]);
		}
		return sqrt(temp);
	}
}

double KMeans::GetLabel(const double* sample, int* label)
{
	// pick the least distance one
	double dist = -1;
	for (int i = 0; i < m_cateNum; i++)
	{
		double temp = CalcDistance(sample, m_means[i], m_dimNum);
		if (temp < dist || dist == -1)
		{
			dist = temp;
			*label = i;
		}
	}
	return dist;
}

void KMeans::Cluster(double *data, int N, int *Label)
{
	int size = N; //sample num
	assert(size >= m_cateNum);

	//Initialize model
	Init(data,size);

	double* x = new double[m_dimNum]; //get one sample
	int label = -1; //class index
	int iterNum = 0;
	double lastCost = 0;
	double currCost = 0;
	int unchanged = 0; //used for 
	bool loop = true;
	int* counts = new int[m_cateNum]; //num of each category
	double** next_means = new double*[m_cateNum]; //centroid of each category
	for(int i = 0; i < m_cateNum; i++)
	{
		next_means[i] = new double[m_dimNum];
	}

	while (loop)
	{
		memset(counts, 0, sizeof(int) * m_cateNum);
		for (int i = 0; i < m_cateNum; i++)
		{
			memset(next_means[i], 0, sizeof(double) * m_dimNum);
		}

		lastCost = currCost;
		currCost = 0;

		//comp loss
		for (int i = 0; i < size; i++)
		{
			for (int j = 0; j < m_dimNum; j++)
				x[j] = data[i * m_dimNum + j]; //get one sample
			currCost += GetLabel(x, &label); //get the label
			
			counts[label]++;

			for (int d = 0; d < m_dimNum; d++)
				next_means[label][d] += x[d];
		}

		currCost /= size;
		
		//update means
		for (int i = 0; i < m_cateNum; i++)
		{
			if (counts[i] > 0)
			{
				for (int d = 0; d < m_dimNum; d++)
				{
					next_means[i][d] /= counts[i];
				}				
				memcpy(m_means[i], next_means[i], sizeof(double) * m_dimNum);
			}
		}	

		//terminal conditions
		iterNum++;
		if (fabs(lastCost - currCost) < m_criteError * lastCost)
		{
			unchanged++;
		}
		if (iterNum > m_maxIter || unchanged > 3)
		{
			loop = false;
		}
		
	} //end while
	
	//end kmeans
	//get label and put into Label
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < m_dimNum; j++)
			x[j] = data[i * m_dimNum + j];
		GetLabel(x, &label);
		Label[i] = label;
	}

	// release memory
	delete[] counts;
	delete[] x;
	for (int i = 0; i < m_cateNum; i++)
		delete[] next_means[i];
	delete[] next_means;
}

void KMeans::Cluster(const char* sampleFileName, const char* labelFileName)
{
	// Check the sample file
	ifstream sampleFile(sampleFileName, ios_base::binary);
	assert(sampleFile);

	int size = 0;
	int dim = 0;
	sampleFile.read((char*)&size, sizeof(int));
	sampleFile.read((char*)&dim, sizeof(int));
	printf("size=%d, dim=%d",size,dim);
	assert(size >= m_cateNum);
	assert(dim == m_dimNum);

	// Initialize model
	Init(sampleFile);

	// Recursion
	double* x = new double[m_dimNum];	// Sample data
	int label = -1;		// Class index
	int iterNum = 0;
	double lastCost = 0;
	double currCost = 0;
	int unchanged = 0;
	bool loop = true;
	int* counts = new int[m_cateNum];
	double** next_means = new double*[m_cateNum];	// New model for reestimation
	for (int i = 0; i < m_cateNum; i++)
	{
		next_means[i] = new double[m_dimNum];
	}

	while (loop)
	{
		memset(counts, 0, sizeof(int) * m_cateNum);
		for (int i = 0; i < m_cateNum; i++)
		{
			memset(next_means[i], 0, sizeof(double) * m_dimNum);
		}

		lastCost = currCost;
		currCost = 0;

		sampleFile.clear();
		sampleFile.seekg(sizeof(int) * 2, ios_base::beg);

		// Classification
		for (int i = 0; i < size; i++)
		{
			sampleFile.read((char*)x, sizeof(double) * m_dimNum);
			currCost += GetLabel(x, &label);

			counts[label]++;
			for (int d = 0; d < m_dimNum; d++)
			{
				next_means[label][d] += x[d];
			}
		}
		currCost /= size;

		// Reestimation
		for (int i = 0; i < m_cateNum; i++)
		{
			if (counts[i] > 0)
			{
				for (int d = 0; d < m_dimNum; d++)
				{
					next_means[i][d] /= counts[i];
				}
				memcpy(m_means[i], next_means[i], sizeof(double) * m_dimNum);
			}
		}

		// Terminal conditions
		iterNum++;
		if (fabs(lastCost - currCost) < m_criteError * lastCost)
		{
			unchanged++;
		}
		if (iterNum >= m_maxIter || unchanged >= 3)
		{
			loop = false;
		}
		//DEBUG
		//cout << "Iter: " << iterNum << ", Average Cost: " << currCost << endl;
	}

	// Output the label file
	ofstream labelFile(labelFileName, ios_base::binary);
	assert(labelFile);

	labelFile.write((char*)&size, sizeof(int));
	sampleFile.clear();
	sampleFile.seekg(sizeof(int) * 2, ios_base::beg);

	for (int i = 0; i < size; i++)
	{
		sampleFile.read((char*)x, sizeof(double) * m_dimNum);
		GetLabel(x, &label);
		labelFile.write((char*)&label, sizeof(int));
	}

	sampleFile.close();
	labelFile.close();

	delete[] counts;
	delete[] x;
	for (int i = 0; i < m_cateNum; i++)
	{
		delete[] next_means[i];
	}
	delete[] next_means;
}


}//end namespace em

#endif
