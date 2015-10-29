#pragma once
#include <iostream>
#include <assert.h>

namespace em{
	//basic class for KMeans GMM HMM
	class EM
	{
	public:
		EM(const int dimNum, const int cateNum){
			m_dimNum = dimNum;
			m_cateNum = cateNum;

			m_maxIter = 100;
			m_criteError = 1e-5;
		}
		//~EM();

		void SetDimNum(int dimNum){m_dimNum = dimNum;}
		void SetcateNum(int cateNum) {m_cateNum = cateNum;}
		void SetCriteError(double criteError){m_criteError = criteError;}
		void SetMaxIter(int maxIter){m_maxIter = maxIter;}

		int GetMaxIter(){return m_maxIter;}
		double GetCriteError(){return m_criteError;}
		int GetCateNum(){return m_cateNum;}
		int GetDimNum(){return m_dimNum;}

	public:
		int m_dimNum; //dimension of samples
		int m_cateNum; //num of category
		int m_maxIter; // The stopping criterion regarding the number of iterations
		double m_criteError; // The stopping criterion regarding the error

	};
}//end namespace em
